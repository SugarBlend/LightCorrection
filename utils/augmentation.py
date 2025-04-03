import cv2
import torch
import random
import numpy as np
import albumentations as A
from typing import List, Tuple


class PatchCutting(A.ImageOnlyTransform):
    def __init__(self, patch_size: Tuple[int, int], p: float = 1.0) -> None:
        super().__init__(p)
        self.patch_size = patch_size

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        h, w, _ = image.shape
        ph, pw = self.patch_size

        patches = []
        for i in range(0, h, ph):
            for j in range(0, w, pw):
                patch = image[i:i + ph, j:j + pw]
                if patch.shape[0] == ph and patch.shape[1] == pw:
                    patches.append(np.transpose(patch, (2, 0, 1)))

        return np.array(patches)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "patch_size",


class AWGN(A.ImageOnlyTransform):
    def __init__(self, sigma_range: Tuple[int, int], p: float = 1.0) -> None:
        super().__init__(p=p)
        self.sigma_range = range(sigma_range[0], sigma_range[1])

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        sigma = random.choices(self.sigma_range, k=1)
        gauss_noise = np.zeros(img.shape, dtype=np.uint8)
        cv2.randn(gauss_noise, np.array([0] * 3), np.array(sigma * 3)).astype(np.uint8)
        gauss_noise = gauss_noise.astype(np.uint8)
        img = cv2.add(img, gauss_noise)
        return img

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "sigma_range",


class ResizeIfSmall(A.ImageOnlyTransform):
    def __init__(self, min_height: int, min_width: int, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.min_height = min_height
        self.min_width = min_width

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        if h < self.min_height or w < self.min_width:
            new_h = max(h, self.min_height)
            new_w = max(w, self.min_width)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return img

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "min_height", "min_width"


class PadIfSmall(A.ImageOnlyTransform):
    def __init__(self, min_height: int, min_width: int, border_value: int = 0, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_value = border_value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w, c = img.shape
        if h < self.min_height or w < self.min_width:
            pad_top = (self.min_height - h) // 2 if h < self.min_height else 0
            pad_bottom = self.min_height - h - pad_top if h < self.min_height else 0
            pad_left = (self.min_width - w) // 2 if w < self.min_width else 0
            pad_right = self.min_width - w - pad_left if w < self.min_width else 0

            img = cv2.copyMakeBorder(
                src=img, top=int(pad_top), bottom=int(pad_bottom), left=int(pad_left), right=int(pad_right),
                borderType=cv2.BORDER_CONSTANT, value=[self.border_value] * c
            )
        return img

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "min_height", "min_width", "border_value"


class NormalizeImage(A.DualTransform):
    def __init__(self, p: float = 1.0) -> None:
        super().__init__(p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img.astype('float32') / 255.0

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()
