"""Checked pixel reads and ROI slicing for the Fusion dataset recipe."""
import cv2


def image(path, flags=cv2.IMREAD_COLOR):
    pixels = cv2.imread(str(path), flags)
    if pixels is None:
        raise ValueError(f'Unreadable {path}')
    return pixels



def crop(im, box):
    x, y, w, h = box
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > im.shape[1] or y+h > im.shape[0]:
        raise ValueError(f'Out-of-bounds ROI {box}')
    return im[y:y+h, x:x+w]
