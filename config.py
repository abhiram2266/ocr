"""Shared settings for generation, training and prediction."""
import string

# Characters your captcha can contain (uppercase letters + digits).
CHARSET = string.ascii_uppercase + string.digits

MIN_LEN, MAX_LEN = 4, 7      # captcha text length range used for synthetic data

IMG_H, IMG_W = 48, 192       # model input size (images are resized to this)

BLANK = 0                    # CTC blank index; real chars are 1..len(CHARSET)
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1

FONTS = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\verdanab.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
]
