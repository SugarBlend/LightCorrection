import cv2
import torch
from pathlib import Path
from dataclasses import dataclass
from model import Model
from dataloader import ColorSpace
from tqdm import tqdm
from torchvision import transforms


@dataclass
class Params:
    kernel_size: int
    model_path: str
    test_dataset_folder: str
    device: str
    color_space: ColorSpace


def eval_correction(config: Params) -> None:
    win_cor = 'Corrected Light'
    cv2.namedWindow(win_cor, cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_KEEPRATIO)
    win_low = 'Low Light'
    cv2.namedWindow(win_low, cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_KEEPRATIO)

    if 'cuda' in config.device and not torch.cuda.is_available():
        config.device = 'cpu'
    device = torch.device(config.device)

    model = Model(kernel_size=config.kernel_size)
    model.load_state_dict(torch.load(config.model_path)['model_state_dict'])
    model.to(device)
    model.eval()

    files = (p.resolve() for p in Path(config.test_dataset_folder).glob("**/*") if
             p.suffix in {".png", ".jpg", ".bmp", ".tiff"})

    for path in tqdm(files, desc='Total'):
        low_image = cv2.imread(str(path))
        if config.color_space.code:
            image = cv2.cvtColor(low_image, config.color_space.code)
        else:
            image = low_image
        normalized_tensor = torch.from_numpy(image).to(device) / 255
        normalized_tensor = normalized_tensor.permute(2, 0, 1)[None]
        std, mean = torch.std_mean(normalized_tensor, dim=[0, 2, 3])
        normalized_tensor = transforms.Compose([transforms.Normalize(mean, std)])(normalized_tensor)

        output = model.forward(normalized_tensor)
        output = (output * 255).clamp(0, 255).round()
        image = output.to(torch.uint8).squeeze().permute(1, 2, 0).cpu().numpy()
        if config.color_space.inverse_code:
            image = cv2.cvtColor(image, config.color_space.inverse_code)
        cv2.imshow(win_low, low_image)
        cv2.imshow(win_cor, image)
        cv2.waitKey(0)


if __name__ == '__main__':
    config = Params(
        kernel_size=3,
        model_path='D:/Projects/Science_test/experiments/2024-07-11-00.29.35/best_psnr.pt',
        test_dataset_folder='D:/Projects/Science_test/LOL-v2/Synthetic/Test/Low',
        device='cuda:0',
        color_space=ColorSpace.RGB
    )

    eval_correction(config)
