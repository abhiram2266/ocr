"""CRNN (CNN + BiLSTM) with CTC decoding, plus shared preprocessing."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import config


def crop_to_box(img: Image.Image) -> Image.Image:
    """Crop to the white text box (both captcha styles draw text on a white box).
    Falls back to the full image if no clear box is found."""
    rgb = img.convert("RGB")
    a = np.asarray(rgb)
    white = a.min(-1) > 225
    rows = np.where(white.mean(1) > 0.3)[0]
    cols = np.where(white.mean(0) > 0.3)[0]
    if len(rows) == 0 or len(cols) == 0:
        return rgb
    y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
    if (y1 - y0) * (x1 - x0) < 0.15 * a.shape[0] * a.shape[1] or (y1 - y0) < 12:
        return rgb
    return rgb.crop((x0, y0, x1 + 1, y1 + 1))


def preprocess(img: Image.Image) -> torch.Tensor:
    """PIL image -> float tensor [1, H, W] in [0,1] (white background ~1)."""
    img = crop_to_box(img).convert("L").resize((config.IMG_W, config.IMG_H), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(a).unsqueeze(0)


def _block(cin, cout, pool):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool) if pool else nn.Identity(),
    )


class CRNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        # 48x192 -> 24x96 -> 12x48 -> 6x48 -> 3x48
        self.cnn = nn.Sequential(
            _block(1, 32, (2, 2)),
            _block(32, 64, (2, 2)),
            _block(64, 128, None),
            _block(128, 128, (2, 1)),
            _block(128, 192, (2, 1)),
        )
        self.rnn = nn.LSTM(192 * 3, 128, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):                      # x: [B,1,H,W]
        f = self.cnn(x)                         # [B,C,3,T]
        b, c, h, t = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)
        f, _ = self.rnn(f)
        return self.fc(f)                       # [B,T,classes]


def greedy_decode(logits: torch.Tensor):
    """logits [B,T,C] -> list of strings (collapse repeats, drop blanks)."""
    out = []
    for seq in logits.argmax(-1).cpu().tolist():
        s, prev = [], None
        for k in seq:
            if k != prev and k != config.BLANK:
                s.append(config.IDX2CHAR[k])
            prev = k
        out.append("".join(s))
    return out
