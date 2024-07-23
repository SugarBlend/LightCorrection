import os
import cv2
import glob
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Any, Tuple, Dict
from enum import Enum
from torchvision import transforms


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
    def __init__(self, data: List[Dict[str, List[str]]], color_space: ColorSpace, device: str = 'cpu',
                 target_transform: Any = None, shape: Tuple[int, int] = (600, 400)):
        self.device: torch.device = torch.device(device)
        self.color_space = color_space
        self.shape = shape
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

        std, mean = torch.std_mean(torch.stack(self._data), dim=[0, 2, 3])
        self.transform = transforms.Compose([transforms.Normalize(mean, std)])
        # self.transform = None

        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, target = self._data[idx], self._targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image.to(self.device), target.to(self.device)

    def prepare_data(self, data: List[str]) -> List[torch.Tensor]:
        packet: List[torch.Tensor] = []
        for item in data:
            image = cv2.imread(item)
            image = cv2.resize(image, self.shape, interpolation=cv2.INTER_LINEAR)
            if self.color_space.code:
                image = cv2.cvtColor(image, self.color_space.code)
            normalized_tensor = torch.from_numpy(image) / 255
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


if __name__ == "__main__":
    root = os.getcwd()

    data = [{'low': [f'{root}/LOL-v2/Synthetic/Train/Low',
                     f'{root}/LOL-v2/Real_captured/Train/Low',],
             'target': [f'{root}/LOL-v2/Synthetic/Train/Normal',
                        f'{root}/LOL-v2/Real_captured/Train/Normal',]}]

    dataset = CustomDataset(data, color_space=ColorSpace.HSV, device='cuda:0')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    for lr, hr in tqdm(data_loader):
        print('\nBatch of images has shape: ', lr.shape, hr.shape)
