import os
import cv2
import glob

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Any, Tuple, Dict, Optional
from enum import Enum


class ColorSpace(Enum):
    RGB = 'rgb'
    HSV = 'hsv'
    BGR = 'bgr'
    YCRCB = 'ycrcb'

    @property
    def code(cls):
        correspondence = {
            cls.RGB: cv2.COLOR_BGR2RGB,
            cls.HSV: cv2.COLOR_BGR2HSV,
            cls.BGR: None,
            cls.YCRCB: cv2.COLOR_BGR2YCrCb
        }
        return correspondence[cls]

    @property
    def inverse_code(cls):
        correspondence = {
            cls.RGB: cv2.COLOR_RGB2BGR,
            cls.HSV: cv2.COLOR_HSV2BGR,
            cls.BGR: None,
            cls.YCRCB: cv2.COLOR_YCrCb2BGR
        }
        return correspondence[cls]


class CustomDataset(Dataset):
    def __init__(
            self,
            data: List[Dict[str, List[str]]],
            color_space: ColorSpace,
            device: str = 'cpu',
            target_transform: Any = None,
            with_crop: bool = False,
            std_mean: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    ) -> None:
        # For test dataset LOLv2 Synthetic + Real captured
        # self.std = torch.tensor([0.1979, 0.2001, 0.2037]).view(1, 3, 1, 1).cuda()
        # self.mean = torch.tensor([0.1269, 0.1372, 0.1364]).view(1, 3, 1, 1).cuda()

        # For train dataset LOLv2 Synthetic + Real captured
        # self.std = torch.tensor([0.2287, 0.2301, 0.2310]).view(1, 3, 1, 1).cuda()
        # self.mean = torch.tensor([0.1524, 0.1637, 0.1711]).view(1, 3, 1, 1).cuda()

        self.with_crop: bool = with_crop
        self.crop_size = np.array([48, 64]) * 3
        self.device: torch.device = torch.device(device)
        self.color_space = color_space

        for struct in data:
            if not all(key in ['low', 'target'] for key in struct.keys()):
                raise KeyError('Every dictionary in data list must have "low" and "target" fields.')

            for paths in struct.values():
                for path in paths:
                    if not os.path.exists(path):
                        raise FileExistsError(f'Path is not exist: {path}')

        low_contrast: List[str] = []
        normal_contrast: List[str] = []
        for struct in data:
            for i in range(len(struct['low'])):
                low_contrast.extend(glob.glob(f'{struct["low"][i]}/*'))
                normal_contrast.extend(glob.glob(f'{struct["target"][i]}/*'))
                assert len(low_contrast) == len(normal_contrast), 'Different length of train files'

        self._data = self.prepare_data(low_contrast)
        self._targets = self.prepare_data(normal_contrast)

        if std_mean is None:
            stds, means = [], []
            for item in self._data:
                std, mean = torch.std_mean(item, dim=[1, 2])
                stds.append(std)
                means.append(mean)
            self.std = torch.stack(stds).mean(dim=0).view(1, 3, 1, 1).cuda()
            self.mean = torch.stack(means).mean(dim=0).view(1, 3, 1, 1).cuda()

            # std, mean = torch.std_mean(torch.stack(self._data), dim=[0, 2, 3])
            # self.transform = transforms.Compose([transforms.Normalize(mean, std)])
            # self.mean = torch.tensor(mean).view(1, 3, 1, 1).cuda()
            # self.std = torch.tensor(std).view(1, 3, 1, 1).cuda()
        else:
            std, mean = std_mean
            self.std = torch.tensor(std).view(1, 3, 1, 1).cuda()
            self.mean = torch.tensor(mean).view(1, 3, 1, 1).cuda()

        self.transform = None

        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, target = self._data[idx], self._targets[idx]

        if self.transform is not None:
            image = self.transform(image)
        image = (image.to(self.device) - self.mean) / self.std

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image.squeeze(0).to(self.device), target.squeeze(0).to(self.device)

    def prepare_data(self, data: List[str]) -> List[torch.Tensor]:
        packet: List[torch.Tensor] = []
        for item in data:
            image = cv2.imread(item)
            if self.color_space.code:
                image = cv2.cvtColor(image, self.color_space.code)
            normalized_tensor = torch.from_numpy(image) / 255

            if self.with_crop:
                h, w, c = normalized_tensor.shape
                height, width = self.crop_size
                for i in range(h // height):
                    for j in range(w // width):
                        crop = normalized_tensor[i * height: i * height + height, j * width: j * width + width]
                        packet.append(crop.permute(2, 0, 1))
            else:
                packet.append(normalized_tensor.permute(2, 0, 1))
        return packet

    @property
    def input_data(self) -> torch.Tensor:
        return self._data

    @property
    def target_data(self) -> torch.Tensor:
        return self._targets

    @target_data.setter
    def target_data(self, value: torch.Tensor) -> None:
        self._targets = value

    @input_data.setter
    def input_data(self, value: torch.Tensor) -> None:
        self._data = value


def default_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    shapes = [item[0].shape for item in batch]
    inputs, targets = [], []
    if len(set(shapes)) > 1:
        _, h, w = min(shapes)
        for low_image, target_image in batch:
            if min(shapes) != low_image.shape:
                _, h_0, w_0 = low_image.shape
                h_0 = h_0 // 2
                w_0 = w_0 // 2
                low_image = low_image[:, h_0 - (h - h//2): h_0 + h//2, w_0 - (w - w//2): w_0 + w//2]
                target_image = target_image[:, h_0 - (h - h//2): h_0 + h//2, w_0 - (w - w//2): w_0 + w//2]
            inputs.append(low_image)
            targets.append(target_image)
    else:
        for low_image, target_image in batch:
            inputs.append(low_image)
            targets.append(target_image)

    return torch.stack(inputs), torch.stack(targets)


if __name__ == "__main__":
    root = os.getcwd()

    data = [{'low': [f'{root}/LOL-v2/Synthetic/Train/Low',
                     f'{root}/LOL-v2/Real_captured/Train/Low',
                     ],
             'target': [f'{root}/LOL-v2/Synthetic/Train/Normal',
                        f'{root}/LOL-v2/Real_captured/Train/Normal',
                        ]}]

    dataset = CustomDataset(data, color_space=ColorSpace.BGR, device='cuda:0')
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=default_collate)
    # dataset = CustomDataset(data, color_space=ColorSpace.BGR, device='cuda:0', with_crop=True)
    # data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    for lr, hr in tqdm(data_loader):
        print('\nBatch of images has shape: ', lr.shape, hr.shape)
