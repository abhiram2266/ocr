# Captcha OCR

A CRNN (CNN + BiLSTM + CTC) that reads the text captchas used on our site.
It handles two styles:

- **bold**: dark-red letters on a white rounded box with grey strike lines.
  This style comes in two variants: tightly packed letters (`QVXKDE`) and
  widely spaced handwriting-style letters (`8F3K9`).
- **plain**: thin coloured letters in a bordered white box on a pale background
  (`SZ6ML`).

Characters: `A-Z` and `0-9`, 4–7 characters long.

## Setup

```
pip install -r requirements.txt
```

The model was trained on CPU. The font paths in `config.py` and `gen.py`
point at `C:\Windows\Fonts`, so generating training data needs Windows.
Reading captchas works on any OS.

## Read a captcha

```
python predict.py samples/QVXKDE.png
python predict.py samples/batch5            # every image in a folder
python predict.py some_folder --errors-only # only the wrong ones, plus accuracy
```

If an image's filename is its answer (`P61OU.png` or `P61OU_3.png`),
`predict.py` reports exact-match accuracy.

From Python:

```python
from predict import read_captcha
text = read_captcha("captcha.png")
```

## Train

```
python train.py                          # from scratch, synthetic data only
python train.py --resume --epochs 10     # continue from checkpoints/crnn.pt
```

Training images are generated on the fly by `gen.py`. To fine-tune on real
captchas, put labeled images in `data/real/` (filename = answer). 20% of them
are held out, and the script prints `real_acc` after each epoch.

To make the held-out synthetic test sets:

```
python gen.py --n 500 --style bold  --seed 7 --out data/test/bold
python gen.py --n 500 --style plain --seed 8 --out data/test/plain
```

## Results (`checkpoints/crnn.pt`)

| Test | Exact match |
|---|---|
| bold, 500 synthetic | 96.8% |
| plain, 500 synthetic | 97.8% |
| real samples (`samples/`) | 9/9 |

Most remaining errors are look-alike characters, mainly `O` vs `0`.
Nine real images is a small sample, so check on a larger labeled set from the
site before relying on these numbers.

## Files

- `config.py`: character set, input size, fonts
- `gen.py`: synthetic captcha generator for both styles
- `model.py`: the CRNN, the crop to the white text box, and CTC decoding
- `train.py`: training loop
- `predict.py`: reading captchas
- `samples/`: real captchas, named after their answers
