"""Train the CRNN.

Synthetic captchas are generated on the fly. If you put real labeled captchas
from your site in data/real/ (filename = label, e.g. QVXKDE.png or QVXKDE_12.png),
they are mixed into training and a held-out slice is used for validation.

    python train.py                       # train from scratch
    python train.py --resume --epochs 10  # continue / fine-tune
"""
import argparse
import glob
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset

import config
import gen
from model import CRNN, greedy_decode, preprocess


def encode(text):
    return [config.CHAR2IDX[c] for c in text]


def label_from_path(p):
    return os.path.splitext(os.path.basename(p))[0].split("_")[0].upper()


class SynthDataset(IterableDataset):
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        random.seed((info.id if info else 0) + random.randint(0, 1 << 30))
        per = self.n // (info.num_workers if info else 1)
        for _ in range(per):
            img, t = gen.sample()
            yield preprocess(img), t


class RealDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        return preprocess(Image.open(p)), label_from_path(p)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    targets = torch.tensor([k for t in texts for k in encode(t)], dtype=torch.long)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    return imgs, targets, lengths, texts


def evaluate(model, loader, device):
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for imgs, _, _, texts in loader:
            preds = greedy_decode(model(imgs.to(device)))
            ok += sum(p == t for p, t in zip(preds, texts))
            tot += len(texts)
    model.train()
    return ok / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per-epoch", type=int, default=20000, help="synthetic samples per epoch")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--real-dir", default="data/real")
    ap.add_argument("--real-repeat", type=int, default=20, help="oversample real images per epoch")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="checkpoints/crnn.pt")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    # Real data split
    real = sorted(glob.glob(os.path.join(a.real_dir, "*.png")) + glob.glob(os.path.join(a.real_dir, "*.jpg")))
    real = [p for p in real if all(c in config.CHARSET for c in label_from_path(p))]
    random.Random(0).shuffle(real)
    n_val = max(1, len(real) // 5) if len(real) >= 5 else 0
    real_val, real_train = real[:n_val], real[n_val:]
    print(f"real images: {len(real)} (train {len(real_train)}, val {len(real_val)})")

    # Fixed per-style validation sets (seeded so numbers are comparable across runs)
    random.seed(1234)
    val_loaders = {}
    for style in gen.STYLES:
        vs = [gen.sample(style) for _ in range(500)]
        val_loaders[style] = DataLoader([(preprocess(i), t) for i, t in vs], batch_size=256, collate_fn=collate)
    random.seed()
    real_val_loader = DataLoader(RealDataset(real_val), batch_size=256, collate_fn=collate) if real_val else None

    model = CRNN().to(device)
    if a.resume and os.path.exists(a.out):
        model.load_state_dict(torch.load(a.out, map_location=device))
        print("resumed from", a.out)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    # Each worker yields per_epoch//workers samples and ends on its own partial batch.
    w = max(a.workers, 1)
    steps_per_epoch = w * -(-(a.per_epoch // w) // a.bs) + -(-(len(real_train) * a.real_repeat) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_per_epoch + 10)
    ctc = nn.CTCLoss(blank=config.BLANK, zero_infinity=True)

    best = -1.0
    for ep in range(1, a.epochs + 1):
        synth = DataLoader(SynthDataset(a.per_epoch), batch_size=a.bs, num_workers=a.workers,
                           collate_fn=collate, persistent_workers=False)
        loaders = [synth]
        if real_train:
            loaders.append(DataLoader(RealDataset(real_train * a.real_repeat), batch_size=a.bs,
                                      shuffle=True, collate_fn=collate))
        total_loss = n = 0
        for loader in loaders:
            for imgs, targets, lengths, _ in loader:
                imgs = imgs.to(device)
                logits = model(imgs)                                   # [B,T,C]
                logp = logits.log_softmax(-1).permute(1, 0, 2)         # [T,B,C]
                in_len = torch.full((imgs.size(0),), logp.size(0), dtype=torch.long)
                loss = ctc(logp, targets, in_len, lengths)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                total_loss += loss.item(); n += 1

        accs = {s: evaluate(model, l, device) for s, l in val_loaders.items()}
        msg = f"epoch {ep:3d}  loss {total_loss / n:.4f}  " + "  ".join(f"{s}_acc {v:.3f}" for s, v in accs.items())
        score = min(accs.values())          # optimise the weaker style, not the average
        if real_val_loader:
            r_acc = evaluate(model, real_val_loader, device)
            msg += f"  real_acc {r_acc:.3f}"
            score = r_acc
        if score >= best:
            best = score
            torch.save(model.state_dict(), a.out)
            msg += "  (saved)"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
