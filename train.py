import glob
import os
import cv2
import traceback
import numpy as np
import torch
import torch.nn as nn
from decimal import Decimal
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Union, NoReturn, Optional
from argparse import Namespace
from colorama import Fore, Style, init
import pytorch_ssim
from enum import Enum
from clearml import Task, Logger
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import kornia
from collections import namedtuple
from model import Model
# from model_1 import Model
# from model_new import Model
from dataloader import CustomDataset, DataLoader, ColorSpace, default_collate
from utils.utils import Timer
from utils.vgg_loss import VGGPerceptualLoss, total_variation_loss, VGGLoss
from utils.pytorch_metrics import SSIM, PSNR, GMSD
from utils.logger import EarlyStopping, TrainingLogger, TestLogger
# from utils.extra_losses import L_color_zy, L_grad_cosist, L_recon, L_exp, L_bright_cosist


class LossType(str, Enum):
    CHARBONNIER = 'charbonnier'
    TV = 'tv'
    PERCEPTUAL = 'perceptual'
    MSE = 'mse'
    SSIM = 'ssim'
    VGG = 'vgg'


def create_optimizer(
        options: Namespace,
        model: torch.nn.Module
) -> Union[torch.optim.Optimizer]:
    import torch.optim as optim
    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if options.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': options.momentum}
    elif options.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (options.beta1, options.beta2),
            'eps': options.epsilon
        }
    elif options.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': options.epsilon}
    else:
        raise argparse.ArgumentError

    kwargs['lr'] = options.lr
    kwargs['weight_decay'] = options.weight_decay

    return optimizer_function(trainable, **kwargs)


def create_scheduler(
        options: Namespace,
        optimizer: torch.optim.Optimizer
) -> Union[torch.optim.lr_scheduler.LRScheduler, NoReturn]:
    from torch.optim import lr_scheduler

    if options.decay_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=options.lr_decay, gamma=options.gamma)
    elif options.decay_type.find('step') >= 0:
        milestones = options.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=options.gamma)
    else:
        raise argparse.ArgumentError

    return scheduler


def save_script_model(model: nn.Module) -> None:
    save_folder = f'{root}/experiments/{clock}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.jit.script(model).save(f'{save_folder}/scripted_model.pt')


class Trainer(object):
    Loss = namedtuple('Loss', [item.value for item in LossType])
    losses = Loss(**{
        LossType.CHARBONNIER.value: kornia.losses.CharbonnierLoss(reduction='sum'),
        LossType.SSIM.value: pytorch_ssim.SSIM(window_size=11),
        LossType.MSE.value: nn.MSELoss(size_average=11),
        LossType.PERCEPTUAL: VGGPerceptualLoss(),
        LossType.VGG: VGGLoss(),
        LossType.TV: total_variation_loss
    })

    def __init__(self, options: argparse.Namespace) -> None:
        self.options = options
        self.device = torch.device(self.options.device)

        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.load_datasets()
        self.model = Model(kernel_size=self.options.kernel_size)
        # self.model = Model(kernel_size=self.options.kernel_size, num_of_layers=10)

        save_script_model(self.model)

        self.optimizer = create_optimizer(args, self.model)
        self.scheduler = create_scheduler(args, self.optimizer)
        self.model.to(self.device)
        self.load_model()

        self.epoch = 0
        self.error_last = 1e8

        self.epoch_timer = Timer()
        self.test_timer = Timer()

        self.train_callback = TrainingLogger(log_interval=10)
        self.early_stopping_callback = EarlyStopping(patience=100)
        self.eval_callback = TestLogger(log_interval=10)

        self.quality_metrics = {
            'psnr': PSNR(),
            'ssim': SSIM(),
            'gmsd': GMSD()
        }
        self.best_eval_stats: Dict[str, Tuple[float, float]] = {name: (0., 0) for name in self.quality_metrics}

    def load_model(self) -> None:
        if os.path.exists(self.options.load_path):
            checkpoint = torch.load(f'{self.options.load_path}')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']

    def load_datasets(self):
        for datatype in ['train', 'test']:
            data = {
                'low': [f'{item}/Low' for item in self.options.__dict__[f'{datatype}_data']],
                'target': [f'{item}/Normal' for item in self.options.__dict__[f'{datatype}_data']]
            }
            self.__dict__[f'{datatype}_dataset'] = CustomDataset([data], self.options.color_space, self.options.device,
                                                                 std_mean=((0.229, 0.224, 0.225), (0.485, 0.456, 0.406)))
            self.__dict__[f'{datatype}_loader'] = DataLoader(
                self.__dict__[f'{datatype}_dataset'], batch_size=self.options.batch_size, shuffle=True, drop_last=False,
                collate_fn=default_collate)

            # self.__dict__[f'{datatype}_dataset'] = CustomDataset(
            #     data=[data], color_space=self.options.color_space, device=self.options.device, with_crop=True,
            #     std_mean=((0.229, 0.224, 0.225), (0.485, 0.456, 0.406)))
            # self.__dict__[f'{datatype}_loader'] = DataLoader(
            #     self.__dict__[f'{datatype}_dataset'], batch_size=self.options.batch_size, shuffle=True, drop_last=False)

    def train(self) -> None:
        self.epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr().pop()

        self.train_callback.on_begin(logs={'epoch': self.epoch, 'lr': lr})
        running_loss = 0.

        self.model.train()

        for batch_idx, (input, target) in enumerate(self.train_loader):
            with self.epoch_timer:
                output = self.model(input)
                loss = 0

                for item in self.options.loss:
                    params = (output,) if item == LossType.TV.value else (output, target)
                    cost = Trainer.losses.__getattribute__(item)(*params)
                    loss += 1 - cost if item == LossType.SSIM.value else cost
                assert isinstance(loss, torch.Tensor)

                if loss.item() < self.options.skip_threshold * self.error_last:
                    for param in self.model.parameters():
                        param.grad = None
                    # self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    print(f'{Fore.LIGHTRED_EX}Skip this batch {batch_idx + 1}! (Loss: {loss.item()}){Style.RESET_ALL}')

            running_loss += loss.item()

            self.train_callback.on_batch_end(batch_idx, logs={
                'batch': (batch_idx + 1),
                'loss': running_loss / (batch_idx + 1),
                'batch time': self.epoch_timer.mean
            })

            if self.options.clearml and batch_idx % args.log_interval == 0:
                series = f"loss: sum({', '.join(self.options.loss)})"
                Logger.current_logger().report_scalar("train", series, value=loss.item(),
                                                      iteration=(self.epoch * len(self.train_loader) + batch_idx))

            self.error_last = loss.item()

        logs = {
            'loss': running_loss / len(self.train_loader),
            'learning rate': Decimal(lr)
        }
        self.train_callback.on_end(self.epoch, logs)

        if self.early_stopping_callback and self.early_stopping_callback.on_epoch_end(self.epoch, logs):
            raise StopIteration

    def freeze_model(self) -> None:
        save_folder = f'{root}/experiments/{clock}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        extra = dict(epoch=self.epoch, model_state_dict=self.model.state_dict(),
                     optimizer_state_dict=self.optimizer.state_dict())
        torch.save(extra, f'{save_folder}/epoch_{self.epoch}.pt')

        if self.epoch == self.best_eval_stats['psnr'][-1]:
            torch.save(extra, f'{save_folder}/best_psnr.pt')

    def test(self) -> None:
        quality: Dict[str, List[Any]] = defaultdict(list)

        self.model.eval()
        self.eval_callback.on_begin()

        with torch.no_grad(), self.test_timer:
            for idx, (image, target) in tqdm(enumerate(self.test_loader)):
                output = self.model(image)
                target = (target * 255).clamp(0, 255).round()
                output = (output * 255).clamp(0, 255).round()
                if self.options.test_show:
                    pos = random.randint(0, len(target) - 1)
                    stack: List[np.ndarray] = []
                    for name, tensor in {'target': target, 'corrected': output}.items():
                        image = tensor[pos].to(torch.uint8).squeeze().permute(1, 2, 0).cpu().numpy()
                        if self.options.color_space.inverse_code:
                            image = cv2.cvtColor(image, self.options.color_space.inverse_code)
                        stack.append(image)
                    cv2.imshow(name, cv2.hconcat(stack))
                    cv2.waitKey(16)

                for metric_name in self.quality_metrics:
                    quality[metric_name].append(self.quality_metrics[metric_name](output[:, :, 3:-3, 3:-3],
                                                                                  target[:, :, 3:-3, 3:-3]).item())

            current_mean_metrics: Dict[str, float] = {}
            for key in quality:
                value = np.mean(quality[key])
                current_mean_metrics[key] = value
                if value > self.best_eval_stats[key][0]:
                    self.best_eval_stats[key] = value, self.epoch

            logs = {
                'current': ', '.join([f'{name}: {value}' for name, value in current_mean_metrics.items()]),
                'best': ', '.join([f'{metric}: {self.best_eval_stats[metric][0]}' for metric in self.best_eval_stats]),
                'elapsed_time': f'{self.test_timer.mean * 1000} ms'
            }
            self.test_timer.reset()
            if self.options.clearml:
                for metric_name in quality.keys():
                    Logger.current_logger().report_scalar("test", metric_name, iteration=self.epoch,
                                                          value=np.mean(quality[metric_name]))
        self.eval_callback.on_end(self.epoch, logs)

        if not self.options.test_only:
            self.freeze_model()

    @torch.no_grad()
    def visualize_hidden_states(self) -> None:
        processed: List[torch.Tensor] = []
        file = random.choice(glob.glob(f'{self.options.test_data[-1]}/Low/*'))
        image = cv2.imread(file)
        if self.options.color_space.code:
            image = cv2.cvtColor(image, self.options.color_space.code)
        image = torch.from_numpy(image).to('cuda:0') / 255
        image = image.permute(2, 0, 1)[None]

        b, c, h, w = image.shape
        padded_x = F.pad(image, pad=[self.options.kernel_size // 2] * 4, mode='reflect')
        sliding_windows = padded_x.unfold(2, self.options.kernel_size, 1).unfold(3, self.options.kernel_size, 1)
        image = sliding_windows.reshape(b, c, h, w, -1)

        for layer in list(self.model.children()):
            image = layer(image)
            processed.append(image.squeeze(0).permute(1, 2, 0, 3).cpu().numpy())

        for i in range(len(processed)):
            feature_maps = (processed[i] * 255).astype(np.uint8)
            features = feature_maps.shape[-1]
            figsize = (8, 15) if features > 1 else (6.4, 4.8)
            fig = plt.figure(figsize=figsize)
            for j in range(features):
                delimiter = min(features, self.options.kernel_size)
                ax = fig.add_subplot(features // delimiter, delimiter, j + 1)
                ax.imshow(feature_maps[..., j])
                ax.axis('off')
                ax.set_title(f'Layer: {i + 1}, Feature: {j + 1}', fontsize=10)
            plt.savefig(f'{root}/experiments/{clock}/{self.epoch}_feature_maps_{i}.png', bbox_inches='tight', dpi=300)

    def terminate(self) -> bool:
        if self.options.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.options.epochs


def parse_opt() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--color_space', type=ColorSpace, default='rgb',
                        help='Colorspace type of input images.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Size of sliding window.')

    parser.add_argument('--train_data', type=str, nargs='+',
                        default=[f'{root}/LOL-v2/Real_captured/Train', f'{root}/LOL-v2/Synthetic/Train'],
                        help='Path to train dataset images.')
    parser.add_argument('--test_data', type=str, nargs='+',
                        default=[f'{root}/LOL-v2/Real_captured/Test', f'{root}/LOL-v2/Synthetic/Test'],
                        help='Path to test dataset images.')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--device', default='cuda:0', help='cuda:0, ... or cpu.')
    parser.add_argument('--skip_threshold', type=float, default=1e2,
                        help='Skipping batch that has large error.')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
    parser.add_argument('--test_only', default=False, help='Set this option to test the model.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Interval for logging (in batch steps).')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--lr_decay', type=int, default=200, help='Learning rate decay per N epochs.')
    parser.add_argument('--decay_type', type=str, default='step', help='Learning rate decay type.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Learning rate decay factor for step decay.')
    parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'),
                        help='Optimizer to use (SGD | ADAM | RMSprop).')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum.')
    parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1.')
    parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2.')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon for numerical stability.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')

    parser.add_argument('--load_path', type=str,
                        default='',
                        help='Path to weights for loading.')
    parser.add_argument('--test_show', type=bool, default=True, help='Show visual correction results.')
    parser.add_argument('--loss', nargs='+', type=LossType,
                        default=['perceptual', 'ssim'],
                        help='Choose loss type.')
    parser.add_argument('--clearml', type=bool, default=False, help='Choose loss type.')
    options = parser.parse_args()
    return options


if __name__ == '__main__':
    from datetime import datetime
    from warnings import filterwarnings

    torch.jit.enable_onednn_fusion(True)
    torch.backends.cudnn.benchmark = True
    random.seed(15236)
    init()
    filterwarnings('ignore')
    clock = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    print(f'{Fore.LIGHTGREEN_EX}Start experiment at: {clock}{Style.RESET_ALL}')
    root = Path(__file__).parent

    args = parse_opt()
    print(f'{Fore.LIGHTGREEN_EX}Color space: {args.color_space.value}, '
          f'kernel size: {args.kernel_size}{Style.RESET_ALL}')

    if args.clearml:
        task = Task.init(project_name='Light correction',
                         task_name=f'color space:{args.color_space}; kernel-size:{args.kernel_size}')
    worker = Trainer(args)
    try:
        while not worker.terminate():
            worker.train()
            worker.test()
            worker.scheduler.step()
            # if (worker.epoch - 1) % 10 == 0:
            #     worker.visualize_hidden_states()
    except StopIteration:
        print(traceback.format_exc())

    if args.clearml:
        task.close()
    print(f'{Fore.LIGHTGREEN_EX}Train finish!{Style.RESET_ALL}')
