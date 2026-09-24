"""Inference-time image preprocessing without albumentations.

Reproduces dataset.get_transforms(input_size, augment=False) (CropWhite(pad=5) -> Resize -> ToGray -> Normalize ->
ToTensorV2) with plain OpenCV/NumPy, so that inference does not import albumentations (and its scikit-image /
scikit-learn dependencies). The output is bitwise identical to the albumentations 1.1.0 pipeline.
"""
import cv2
import numpy as np
import torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def crop_white(img, value=(255, 255, 255), pad=0):
    height, width, _ = img.shape
    x = (img != value).sum(axis=2)
    top, bottom, left, right = 0, height, 0, width
    if x.sum() != 0:
        rows = np.flatnonzero(x.sum(axis=1))
        cols = np.flatnonzero(x.sum(axis=0))
        top, bottom = rows[0], rows[-1] + 1
        left, right = cols[0], cols[-1] + 1
    img = img[top:bottom, left:right]
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=value)


class InferenceTransform:

    def __init__(self, input_size, pad=5):
        self.input_size = input_size
        self.pad = pad
        mean = np.array(MEAN, dtype=np.float32) * 255.0
        std = np.array(STD, dtype=np.float32) * 255.0
        self.mean = mean
        self.denominator = np.reciprocal(std, dtype=np.float32)

    def __call__(self, image):
        img = crop_white(image, pad=self.pad)
        if img.shape[:2] != (self.input_size, self.input_size):
            img = cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        img = img.astype(np.float32)
        img -= self.mean
        img *= self.denominator
        return torch.from_numpy(img.transpose(2, 0, 1))
