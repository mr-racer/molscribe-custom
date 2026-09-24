"""Inference-time image preprocessing without albumentations.

Reproduces dataset.get_transforms(input_size, augment=False) (CropWhite(pad=5) -> Resize -> ToGray -> Normalize ->
ToTensorV2) with plain OpenCV/NumPy, so that inference does not import albumentations (and its scikit-image /
scikit-learn dependencies). The output is bitwise identical to the albumentations 1.1.0 pipeline.

The work is split in two so that batches can be prepared in threads while the GPU is busy:
`prepare` (CPU, OpenCV releases the GIL) returns the resized grayscale uint8 image, `normalize` turns a stack of
them into the float model input on any device.
"""
import cv2
import numpy as np
import torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WHITE = (255, 255, 255)


def crop_white(img, pad=0):
    """Crop to the bounding box of non-white pixels (any channel != 255), then pad with white."""
    non_white = cv2.bitwise_not(cv2.inRange(img, WHITE, WHITE))
    x, y, w, h = cv2.boundingRect(non_white)
    if w > 0 and h > 0:
        img = img[y:y + h, x:x + w]
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=WHITE)


class InferenceTransform:

    def __init__(self, input_size, pad=5):
        self.input_size = input_size
        self.pad = pad
        self.mean = torch.tensor(MEAN, dtype=torch.float32) * 255.0
        std = np.array(STD, dtype=np.float32) * 255.0
        self.denominator = torch.from_numpy(np.reciprocal(std, dtype=np.float32))

    def prepare(self, image):
        """RGB uint8 image -> (input_size, input_size) grayscale uint8."""
        img = crop_white(image, pad=self.pad)
        if img.shape[:2] != (self.input_size, self.input_size):
            img = cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def normalize(self, gray):
        """(B, H, W) uint8 tensor -> (B, 3, H, W) float32, same arithmetic as albumentations' Normalize."""
        x = gray.unsqueeze(1).expand(-1, 3, -1, -1).float()
        x = x - self.mean.to(x.device).view(1, 3, 1, 1)
        return x * self.denominator.to(x.device).view(1, 3, 1, 1)

    def __call__(self, image):
        return self.normalize(torch.from_numpy(self.prepare(image)).unsqueeze(0))[0]
