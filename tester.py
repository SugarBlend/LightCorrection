import random

import mlflow
import json
import cv2
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm
import torch
from typing import Tuple, Optional, List, Dict, Callable
from torch.utils.data.dataloader import DataLoader
from utils.pytorch_metrics import PSNR, GMSD, LPIPS, SSIM
from utils.union_dataloader import UnionDataset, default_collate
import albumentations as A
from utils.augmentation import AWGN


def plot_graphs(images: List[Tuple[int, List[np.ndarray]]], bias: float = 0.75, global_title: str = "") -> None:
    font_dict = {'fontname': 'Comic Sans MS', 'fontsize': 14}
    titles = ["Входное изображение", "Результирующее изображение", "Эталонное изображение"]

    for instance in images:
        frame_number, (inputs, outputs, targets) = instance
        inputs, outputs, targets = [item.squeeze(0).transpose((1, 2, 0)).astype(np.float32)
                                    for item in [inputs, outputs, targets]]

        cols = targets.shape[1]
        row = targets.shape[0] // 2
        x_range = np.arange(cols)

        fig, axes = plt.subplots(4, 3, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 1, 1, 1]})
        fig.suptitle(global_title, **font_dict, y=1)

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        for i, (img, title) in enumerate(zip([inputs, outputs, targets], titles)):
            img = cv2.line(np.ascontiguousarray(img).astype(np.uint8), (0, row), (cols - 1, row), (255, 255, 255), 8)
            axes[0, i].imshow(img)
            axes[0, i].set_title(title, **font_dict)
            axes[0, i].axis('off')

        def plot_slice_1d(ax: matplotlib.axes.Axes, channel: int, title: str) -> None:
            small_font = {'fontname': 'Comic Sans MS', 'fontsize': 12}
            ax.plot(x_range, inputs[row, :, channel], 'p--', marker='*', markerfacecolor='green', alpha=0.6,
                    label="Входное", linestyle="dashed", markersize=3)
            ax.plot(x_range, outputs[row, :, channel] + bias, 'bo-', marker='h', markerfacecolor='black', alpha=0.6,
                    label=f"Выходное (+{bias})", markersize=3)
            ax.plot(x_range, targets[row, :, channel] + 2 * bias, 'o--', marker='o', alpha=0.6,
                    label=f"Эталон (+{bias * 2})", markersize=3)
            ax.set_title(title, **small_font)
            ax.set_ylim(-0.2, 1 + 2 * bias)
            ax.set_xlim(0, cols)
            ax.legend()
            ax.grid()
            ax.set_xticks(np.linspace(0, cols - 1, num=10, dtype=int))
            ax.set_xlabel("Пиксель", **small_font)
            ax.set_ylabel("Нормированная интенсивность", **small_font)

        for img in [inputs, outputs, targets]:
            if img.max() > 1:
                img /= 255.0

        for i, color in enumerate(["R", "G", "B"], start=2):
            plot_slice_1d(fig.add_subplot(4, 1, i), i - 2, f"Анализ среза (Канал {color})")

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show(block=False)

        mlflow.log_figure(fig, artifact_file=f"tester/{global_title}/{str(frame_number).rjust(5, '0')}.jpg")
        plt.close()


def run_pipeline(
        model: torch.nn.Module,
        loader: DataLoader,
        normalization_range: Tuple[int, int],
        device: str,
        samples: Optional[List[int]] = None,
        using_metrics: List[Callable] = [PSNR, SSIM, GMSD, LPIPS]
) -> Tuple[Dict[str, float], List[Tuple[int, List[np.ndarray]]]]:
    device = torch.device(device)
    images: List[Tuple[int, List[np.ndarray]]] = []
    metrics = {metric.__name__: metric() for metric in using_metrics}
    total_quality = {metric: [] for metric in metrics}

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc='Evaluation step')):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if normalization_range == [-1, 1]:
            inputs = inputs * 2 - 1

        outputs = model(inputs)

        if normalization_range == [-1, 1]:
            inputs, outputs = [((item + 1) * 128).clip(0, 255) for item in [inputs, outputs]]
            targets = targets.clip(0, 1) * 255
        else:
            inputs, outputs, targets = [item.clip(0, 1) * 255 for item in [inputs, outputs, targets]]

        if isinstance(samples, List):
            always = True if not len(samples) else batch_idx in samples
        else:
            always = False

        if always:
            images.append((batch_idx, [item.to(torch.uint8).cpu().numpy() for item in [inputs, outputs, targets]]))

        for metric in metrics:
            total_quality[metric].append(metrics[metric](outputs.float(), targets.float()))

    total_quality = {metric: np.mean(list(filter(lambda x: x < np.inf, total)))
                     for metric, total in total_quality.items()}

    return total_quality, images


def tune_tracking(root: str, run_id: str, uri: str, task: str) -> None:
    model = mlflow.pytorch.load_model(uri)
    model.eval().cuda()

    json_str = mlflow.artifacts.load_text(f"runs:/{run_id}/configs/experiment.json")
    json_data = json.loads(json_str)
    device = json_data["training"]["device"]

    json_str = mlflow.artifacts.load_text(f"runs:/{run_id}/configs/dataset_config.json")
    json_data = json.loads(json_str)
    normalization_range = json_data["normalization_range"]

    if task == 'light_correction':
        dataset = UnionDataset(
            data={
                "low": [f"{root}/LOL-v2/Synthetic/Test/Low", f"{root}/LOL-v2/Real_captured/Test/Low"],
                "target": [f"{root}/LOL-v2/Synthetic/Test/Normal", f"{root}/LOL-v2/Real_captured/Test/Normal"],
            }
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=default_collate)
        samples = random.choices(range(len(loader)), k=3)
        quality, images = run_pipeline(model, loader, normalization_range, device, samples)
        global_title = 'Light correction'
        mlflow.log_dict(quality, f'tester/{global_title}/metrics.json')
        plot_graphs(images, global_title=global_title)

    elif task == 'denoising':
        dataset = UnionDataset(
            data={
                "low": [f"{root}/LOL-v2/Synthetic/Train/Normal", f"{root}/LOL-v2/Real_captured/Train/Normal"],
                "target": [f"{root}/LOL-v2/Synthetic/Train/Normal", f"{root}/LOL-v2/Real_captured/Train/Normal"],
            }
        )
        for sigma in range(5, 26, 5):
            dataset.inputs_transform = A.Compose([
                AWGN(sigma_range=(sigma, sigma + 1), p=1.0)
            ])
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=default_collate)
            samples = random.choices(range(len(loader)), k=3)

            quality, images = run_pipeline(model, loader, normalization_range, device, samples)
            global_title = f'AWGN_\u03BC_0_\u03C3_{sigma}'
            mlflow.log_dict(quality, f'tester/{global_title}/metrics.json')
            plot_graphs(images, global_title=global_title)
    else:
        raise ValueError


if __name__ == '__main__':
    random.seed(65)
    root = Path(os.getcwd()).parent

    # task = 'light_correction'
    task = 'denoising'
    checkpoint_name = "best_model_epoch_39"
    run_id = "f0c3daf22c03438bb6f08a5540a25730"

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    uri = f"runs:/{run_id}/checkpoints/{checkpoint_name}"
    with mlflow.start_run(run_id=run_id):
        tune_tracking(root, run_id, uri, task)
