import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
# from torchvision import transforms
from collections import defaultdict
from typing import List, Optional, Tuple, Any, Dict, Union

from model import Model
# from model_new import Model
from dataloader import ColorSpace
from utils.metrics import PSNR, SSIM


@dataclass
class Params:
    kernel_size: int
    model_path: str
    test_dataset_folder: str
    target_dataset_folder: str
    device: str
    color_space: ColorSpace
    rayleigh_noise: bool = False
    noise_std_range: Optional[range] = None

    def description(self) -> str:
        return f'{self.color_space.value}_{self.kernel_size}'


def correction(
        config: Params,
        data: Dict[str, List[float | str]],
        show_collage: bool,
        calculate_metrics: bool
) -> None:
    if 'cuda' in config.device and not torch.cuda.is_available():
        config.device = 'cpu'
    device = torch.device(config.device)

    model = Model(kernel_size=config.kernel_size)
    # model = Model(kernel_size=config.kernel_size, num_of_layers=10)
    model.load_state_dict(torch.load(config.model_path)['model_state_dict'])
    model.to(device)
    model.eval()

    files = (p.resolve() for p in Path(config.test_dataset_folder).glob("**/*") if
             p.suffix in {".png", ".jpg", ".bmp", ".tiff"})
    files = list(files)

    if show_collage:
        results = {}
        if isinstance(config.noise_std_range, range):
            for sigma in config.noise_std_range:
                images, scores = run_pipeline(model, files, device, distribution={'normal': sigma})
                results[sigma] = images
        elif config.rayleigh_noise:
            images, scores = run_pipeline(model, files, device, distribution={'rayleigh': ...})
            results['rayleigh'] = images
        else:
            images, scores = run_pipeline(model, files, device)
            results['none'] = images

        plot_results(results, config.color_space.value)

    if calculate_metrics:
        def calculate_quality(iterator: Union[range, List[None]] = [None], extra_meta: str = '',
                              distribution: Dict[str, Any] = {}) -> None:
            for sigma in iterator:
                if sigma is not None:
                    distribution = {'normal': sigma}
                _, scores = run_pipeline(model, files, device, calculate_metrics=True, distribution=distribution)
                appendix = extra_meta.format(sigma=sigma) if extra_meta else extra_meta
                data['Метод'].append(f'{str(config.color_space.value).upper()}, {config.kernel_size}x{config.kernel_size}'
                                     f'{appendix}')
                data['PSNR'].append(np.mean(scores['psnr']))
                data['SSIM'].append(np.mean(scores['ssim']))
                data['GMSD'].append(np.mean(scores['gmsd']))

        if isinstance(config.noise_std_range, range):
            calculate_quality(config.noise_std_range, ', Нормальное распределение, \u03BC = 0, \u03C3 = {sigma}')
        elif config.rayleigh_noise:
            calculate_quality(extra_meta=', распределение Релея, \u03C3 = 0.27',  distribution={'rayleigh': ...})
        else:
            calculate_quality(extra_meta=', без шума')


def run_pipeline(
        model: torch.nn.Module,
        files: List[Path],
        device: torch.device,
        calculate_metrics: bool = False,
        distribution: Dict[str, Any] = {}
) -> Tuple[List[np.ndarray], defaultdict[Any, List]]:
    if not calculate_metrics:
        positions = [4, 67, 9]
        files = np.array(files)[positions]

    images: List[np.ndarray] = []
    results = defaultdict(list)
    for path in tqdm(files, desc='Processed frame'):
        image = cv2.imread(str(path))

        if 'normal' in distribution:
            gauss = np.random.normal(0, distribution['normal'], image.shape)
            image = image + gauss

        elif 'rayleigh' in distribution:
            image = add_speckle_noise(image)

        low_image = image.astype(np.uint8)

        if config.color_space.code:
            image = cv2.cvtColor(image.astype(np.uint8), config.color_space.code)

        normalized_tensor = torch.from_numpy(image).to(device) / 255
        normalized_tensor = normalized_tensor.permute(2, 0, 1)[None].to(torch.float32)

        # For train dataset LOLv2 Synthetic + Real captured
        # std = torch.tensor([0.2287, 0.2301, 0.2310]).view(1, 3, 1, 1).cuda()
        # mean = torch.tensor([0.1524, 0.1637, 0.1711]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.1431, 0.1447, 0.1493]).view(1, 3, 1, 1).cuda()
        mean = torch.tensor([0.1526, 0.1639, 0.1713]).view(1, 3, 1, 1).cuda()
        normalized_tensor = (normalized_tensor.to(device) - mean) / std

        output = model.forward(normalized_tensor)
        output = (output * 255).clamp(0, 255).round()
        image = output.to(torch.uint8).squeeze().permute(1, 2, 0).cpu().numpy()
        if config.color_space.inverse_code:
            image = cv2.cvtColor(image, config.color_space.inverse_code)

        target_path = str(path).replace('Low', 'Normal').replace('low', 'normal')
        target = cv2.imread(target_path)
        images.extend([low_image, image, target])
        if calculate_metrics:
            image = image.astype(np.float32)
            target = target.astype(np.float32)
            results['psnr'].append(PSNR()(image, target))
            results['ssim'].append(SSIM()(image, target))
            results['gmsd'].append(np.mean(cv2.quality.QualityGMSD().compute(image, target)[0][:-1]))

    return images, results


def plot_results(results: Dict[int, List[np.ndarray]], colorspace: str) -> None:
    for sheet in results:
        fig, axs = plt.subplots(len(results[sheet]) // 3, 3, figsize=(12, 12))  # [input, output, target] x n
        if isinstance(sheet, int):
            title = f'Результаты при добавлении шума с параметрами:  \u03BC = 0, \u03C3 = {sheet}'
        elif sheet == 'none':
            title = 'Результаты без добавления шума'
        elif sheet == 'rayleigh':
            title = 'Результаты c добавлением спекл-шума с параметрами: \u03C3 = 0.27'
        fig.suptitle(title, fontname='Comic Sans MS', fontsize=16)

        font_dict = {'fontname': 'Comic Sans MS', 'fontsize': 14}
        axs[0, 0].set_title(f'Входные данные, формат {colorspace}', font_dict)
        axs[0, 1].set_title('Результат коррекции, формат rgb', font_dict)
        axs[0, 2].set_title('Целевое изображение, формат rgb', font_dict)
        axs = axs.flatten()
        for img, ax in zip(results[sheet], axs):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fig.tight_layout()
        plt.show(block=False)

        base_folder = f'./evaluations/{timestamp}'
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        print(f'{base_folder}/{config.description()}_sigma-{sheet}.png')
        fig.savefig(f'{base_folder}/{config.description()}_sigma-{sheet}.png', dpi=300)
        plt.close(fig)


def add_speckle_noise(img: np.ndarray, scale: float = 0.2707) -> np.ndarray:
    img = img.astype(np.float32)
    avg = np.mean(img)
    size = img.shape
    noise = np.random.rayleigh(scale=scale, size=size)

    noised_img = img + img * noise - avg * scale
    noised_img = np.where(noised_img <= 255, noised_img, 255)
    return noised_img


if __name__ == '__main__':
    cv2.ocl.setUseOpenCL(False)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    base_folder = 'C:/Users/Alexander/Downloads/Telegram Desktop/experiments/experiments/'
    experiments = [
        dict(
            # model_path='D:/Projects/LightCorrection/experiments/2024-10-12-22.38.47/best_psnr.pt',
            model_path='D:/Projects/LightCorrection/experiments/2024-11-06-17.52.41/best_psnr.pt',
            kernel_size=3, color_space=ColorSpace.BGR
        ),
        dict(
            # model_path='D:/Projects/LightCorrection/experiments/2024-10-12-22.38.47/best_psnr.pt',
            model_path='D:/Projects/LightCorrection/experiments/2024-11-06-17.52.41/best_psnr.pt',
            kernel_size=3, color_space=ColorSpace.BGR,
            rayleigh_noise=True
        ),
        dict(
            # model_path='D:/Projects/LightCorrection/experiments/2024-10-12-22.38.47/best_psnr.pt',
            model_path='D:/Projects/LightCorrection/experiments/2024-11-06-17.52.41/best_psnr.pt',
            kernel_size=3, color_space=ColorSpace.BGR,
            noise_std_range=range(5, 26, 5)
        ),
        # dict(
        #     model_path=f'D:/Projects/LightCorrection/experiments/2024-10-09-00.05.18/best_psnr.pt',
        #     kernel_size=3, color_space=ColorSpace.BGR
        # ),
        # dict(
        #     model_path=f'{base_folder}/2024-07-27-01.37.44/best_psnr.pt',
        #     kernel_size=3, color_space=ColorSpace.HSV
        # ),
        # dict(
        #     model_path=f'{base_folder}/2024-07-27-11.18.17/best_psnr.pt',
        #     kernel_size=3, color_space=ColorSpace.BGR
        # ),
        # dict(
        #     model_path=f'{base_folder}/2024-07-27-20.57.11/best_psnr.pt',
        #     kernel_size=3, color_space=ColorSpace.YCRCB
        # ),
        # dict(
        #     model_path=f'{base_folder}/2024-07-28-06.32.52/best_psnr.pt',
        #     kernel_size=3, color_space=ColorSpace.RGB
        # ),
        # dict(
        #     model_path=f'{base_folder}/2024-07-28-16.10.44/best_psnr.pt',
        #     kernel_size=5, color_space=ColorSpace.HSV
        # ),
        # dict(
        #     model_path=f'{base_folder}/2024-07-29-16.04.22/best_psnr.pt',
        #     kernel_size=5, color_space=ColorSpace.RGB
        # ),
        # dict(
        #     model_path=f'{base_folder}/2024-07-31-11.48.42/best_psnr.pt',
        #     kernel_size=5, color_space=ColorSpace.YCRCB
        # ),
        #
        # dict(
        #     model_path=f'{base_folder}/2024-08-01-11.47.26/best_psnr.pt',
        #     kernel_size=7, color_space=ColorSpace.HSV
        # ),
        # dict(
        #     model_path=f'{base_folder}/2024-08-03-12.08.10/best_psnr.pt',
        #     kernel_size=7, color_space=ColorSpace.RGB
        # ),
        # dict(
        #     model_path=f'{base_folder}/2024-08-05-12.25.30/best_psnr.pt',
        #     kernel_size=7, color_space=ColorSpace.YCRCB
        # )
    ]

    show_collage = True
    calculate_metrics = True

    data = defaultdict(list)
    for experiment_meta in tqdm(experiments, desc='Total'):
        config = Params(
            **experiment_meta,
            test_dataset_folder='D:/Projects/LightCorrection/LOL-v2/Real_captured/Test/Low',
            target_dataset_folder='D:/Projects/LightCorrection/LOL-v2/Real_captured/Test/Normal',
            device='cuda:0'
        )
        correction(config, data, show_collage, calculate_metrics)

    if calculate_metrics:
        base_folder = f'./evaluations/{timestamp}'
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        df = pd.DataFrame(data)
        df.to_excel(f'{base_folder}/metrics_{timestamp}.xlsx', index=False)
