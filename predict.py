"""Read captcha images.

    python predict.py samples/QVXKDE.png
    python predict.py data/real            # whole folder; reports accuracy if filenames are labels
"""
import argparse
import glob
import os

import torch
from PIL import Image

from model import CRNN, greedy_decode, preprocess

_model = None


def load(ckpt="checkpoints/crnn.pt", device=None):
    global _model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = CRNN().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    _model = (m, device)
    return _model


def read_captcha(img) -> str:
    """img: path or PIL.Image -> predicted text."""
    m, device = _model or load()
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    with torch.no_grad():
        return greedy_decode(m(preprocess(img).unsqueeze(0).to(device)))[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ckpt", default="checkpoints/crnn.pt")
    ap.add_argument("--errors-only", action="store_true", help="only print wrong predictions")
    a = ap.parse_args()
    load(a.ckpt)
    paths = sorted(glob.glob(os.path.join(a.path, "*.*"))) if os.path.isdir(a.path) else [a.path]
    ok = 0
    for p in paths:
        pred = read_captcha(p)
        truth = os.path.splitext(os.path.basename(p))[0].split("_")[0].upper()
        ok += pred == truth
        if not a.errors_only or pred != truth:
            print(f"{os.path.basename(p):30s} -> {pred}" + ("" if pred == truth else f"   (expected {truth})"))
    if len(paths) > 1:
        print(f"\nexact-match accuracy (if filenames are labels): {ok}/{len(paths)} = {ok / len(paths):.1%}")
