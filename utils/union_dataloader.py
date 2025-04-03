import os
import cv2
import glob
import torch
import albumentations as A
from typing import Dict, Optional, Tuple, List
from torch.utils.data import DataLoader, Dataset
from utils.augmentation import NormalizeImage


def default_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    min_h = min(item[0].shape[1] for item in batch)
    min_w = min(item[0].shape[2] for item in batch)

    inputs = torch.stack([low[:, :min_h, :min_w] for low, _ in batch])
    targets = torch.stack([target[:, :min_h, :min_w] for _, target in batch])

    return inputs, targets


class UnionDataset(Dataset):
    def __init__(
            self,
            data: Dict[str, List[str]],
            patch_size: Tuple[int, int] = (64, 64),
            inputs_transform: Optional[A.Compose] = None,
            both_transform: Optional[A.ReplayCompose] = None,
            device: str = 'cuda:0'
    ) -> None:
        self.patch_size: Tuple[int, int] = patch_size
        self.inputs_transform: Optional[A.Compose] = inputs_transform
        self.both_transform: Optional[A.ReplayCompose] = both_transform
        self.device: torch.device = torch.device(device)

        if not all(k in data for k in ["low", "target"]):
            raise KeyError('Data dictionary must contain "low" and "target" fields.')

        self.inputs_paths = sorted([p for d in data["low"] for p in glob.glob(f"{d}/*")])
        self.targets_paths = sorted([p for d in data["target"] for p in glob.glob(f"{d}/*")])

        if len(self.inputs_paths) != len(self.targets_paths):
            raise ValueError("Mismatch between low and target dataset sizes.")

        if self.both_transform is None:
            self.both_transform = A.ReplayCompose([
                NormalizeImage(p=1.0),
                A.ToTensorV2()
            ])
        else:
            assert isinstance(self.both_transform, A.ReplayCompose)
            self.both_transform = both_transform

    def __len__(self) -> int:
        return len(self.inputs_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = cv2.cvtColor(cv2.imread(self.inputs_paths[idx]), cv2.COLOR_BGR2RGB)
        targets = cv2.cvtColor(cv2.imread(self.targets_paths[idx]), cv2.COLOR_BGR2RGB)

        if self.inputs_transform:
            inputs = self.inputs_transform(image=inputs)["image"]

        if self.both_transform:
            results = self.both_transform(image=inputs)
            targets = A.ReplayCompose.replay(results["replay"], image=targets)["image"]
            inputs = results["image"]

        return inputs.to(self.device), targets.to(self.device)


if __name__ == "__main__":
    root = os.getcwd()
    dataset = UnionDataset(
        data={
            "low": [f"{root}/LOL-v2/Synthetic/Train/Normal", f"{root}/LOL-v2/Real_captured/Train/Normal"],
            "target": [f"{root}/LOL-v2/Synthetic/Train/Normal", f"{root}/LOL-v2/Real_captured/Train/Normal"],
        },
        inputs_transform=A.Compose([
            A.GaussNoise(std_range=(5 / 255, 25 / 255), p=1.0)
        ])
    )
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=default_collate)

    for lr, hr in data_loader:
        lr = lr.clamp(0, 1) * 255
        hr = hr.clamp(0, 1) * 255
        combined = cv2.hconcat([
            lr[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy(),
            hr[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        ])
        cv2.imshow("Visualization", combined)
        cv2.waitKey(0)
        print("Batch shape:", lr.shape, hr.shape)
