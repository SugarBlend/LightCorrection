import numpy as np
import cv2
from typing import List, Tuple
from scipy import signal


class PSNR(object):
    def __init__(self):
        self.name = self.__class__.__name__

    @staticmethod
    def __call__(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        return 20 * np.log10(255.0 / np.sqrt(mse))


class SSIM(object):
    def __init__(self):
        self.name = self.__class__.__name__

    @staticmethod
    def _ssim(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims: List[np.ndarray] = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")


class GMSD(object):
    def __init__(self, channels: int = 3):
        self.name = self.__class__.__name__
        self.channels: int = channels
        dx = (np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.)[None]
        self.dx = np.tile(dx, (3, 1, 1))

        dy = (np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3.)[None]
        self.dy = np.tile(dy, (3, 1, 1))
        self.aveKernel = np.ones((self.channels, 1, 2, 2)) / 4.

    def __call__(self, img1: np.ndarray, img2: np.ndarray,  T: int = 170) -> float:
        y1 = signal.convolve2d(img1, self.aveKernel, mode='same', boundary='symm')
        y2 = signal.convolve2d(img2, self.aveKernel, mode='same', boundary='symm')

        ixy1 = signal.convolve2d(y1, self.dx, mode='same', boundary='symm')
        iyy1 = signal.convolve2d(y1, self.dy, mode='same', boundary='symm')
        gradient_map1 = np.sqrt(ixy1 ** 2 + iyy1 ** 2 + 1e-12)

        ixy2 = signal.convolve2d(y2, self.dx, mode='same', boundary='symm')
        iyy2 = signal.convolve2d(y2, self.dy, mode='same', boundary='symm')
        gradient_map2 = np.sqrt(ixy2 ** 2 + iyy2 ** 2 + 1e-12)

        quality_map = (2 * gradient_map1 * gradient_map2 + T) / (gradient_map1 ** 2 + gradient_map2 ** 2 + T)
        score = np.std(quality_map)
        return score
