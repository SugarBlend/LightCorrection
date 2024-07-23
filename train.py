import os
import cv2
import traceback
import numpy as np
import torch
import torch.nn as nn
from torch.optim import RMSprop, Adam, SGD
from decimal import Decimal
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from argparse import Namespace
from colorama import Fore, Style, init
import pytorch_ssim
from enum import Enum
from clearml import Task, Logger
import random
from tqdm import tqdm

from model import Model
from dataloader import CustomDataset, DataLoader, ColorSpace
from utils.utils import Timer
from utils.vgg_loss import VGGPerceptualLoss
from utils.pytorch_metrics import SSIM, MS_SSIM, PSNR
from utils.logger import EarlyStopping, TrainingLogger, TestLogger


class LossType(str, Enum):
    COMBINE = 'combine'
    PERCEPTUAL = 'perceptual'
    MSE = 'mse'
    SSIM = 'ssim'


def create_optimizer(args: Namespace, model: torch.nn.Module) -> RMSprop | Adam | SGD:
    import torch.optim as optim
    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}
    else:
        raise argparse.ArgumentError

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def create_scheduler(args: Namespace, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    from torch.optim import lr_scheduler

    if args.decay_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    else:
        raise argparse.ArgumentError

    return scheduler


class Trainer(object):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(self.args.device)

        low_data = [f'{item}/Low' for item in self.args.train_data]
        target_data = [f'{item}/Normal' for item in self.args.train_data]
        train_dataset = CustomDataset(data=[dict(low=low_data, target=target_data)],
                                      color_space=args.color_space, device=self.args.device)
        self.loader_train = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False)

        low_data = [f'{item}/Low' for item in self.args.test_data]
        target_data = [f'{item}/Normal' for item in self.args.test_data]
        test_dataset = CustomDataset(data=[dict(low=low_data, target=target_data)],
                                     color_space=args.color_space, device=self.args.device)
        self.loader_test = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

        self.model = Model(kernel_size=self.args.kernel_size)

        self.criterion_ssim = pytorch_ssim.SSIM(window_size=11)
        self.criterion_mse = nn.MSELoss(size_average=True)
        self.perceptual_loss = VGGPerceptualLoss()

        self.optimizer = create_optimizer(args, self.model)
        self.scheduler = create_scheduler(args, self.optimizer)
        self.model.to(self.device)
        self.model.train()

        self.epoch = 0
        self.error_last = 1e8

        if os.path.exists(self.args.load_path):
            checkpoint = torch.load(f'{self.args.load_path}')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.criterion = checkpoint['loss']

        # logging and measures
        self.epoch_timer = Timer()
        self.test_timer = Timer()

        self.train_callback = TrainingLogger(log_interval=10)
        self.early_stopping_callback = EarlyStopping(patience=100)
        self.eval_callback = TestLogger(log_interval=10)

        self.similarity = [PSNR(), MS_SSIM()]
        self.qualities = defaultdict(list)
        self.eval_stats: Dict[str, Tuple[float, float]] = {item.name: (0., 0) for item in self.similarity}

    def train(self) -> None:
        self.epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr().pop()

        self.train_callback.on_begin(logs={'epoch': self.epoch, 'lr': lr})
        running_loss = 0.

        self.model.train()
        self.optimizer.step()
        self.scheduler.step()

        for batch_idx, (input, target) in enumerate(self.loader_train):
            with (self.epoch_timer):
                self.optimizer.zero_grad()
                output = self.model(input)

                if self.args.loss == LossType.COMBINE:
                    loss = (self.criterion_ssim(output, target) + self.perceptual_loss(output, target))
                elif self.args.loss == LossType.PERCEPTUAL:
                    loss = self.perceptual_loss(output, target)
                elif self.args.loss == LossType.MSE:
                    loss = self.criterion_mse(output, target)
                elif self.args.loss == LossType.SSIM:
                    loss = self.criterion_ssim(output, target)

                if loss.item() < self.args.skip_threshold * self.error_last:
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

            if batch_idx % args.log_interval == 0:
                Logger.current_logger().report_scalar("train", "loss", value=loss.item(),
                                                      iteration=(self.epoch * len(self.loader_train) + batch_idx))

            self.error_last = loss.item()

        logs = {
            'loss': running_loss / len(self.loader_train),
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

        if self.epoch == self.eval_stats['PSNR'][-1]:
            torch.save(extra, f'{save_folder}/best_psnr.pt')

    def test(self) -> None:
        quality: Dict[str, List[Any]] = defaultdict(list)

        self.model.eval()
        self.eval_callback.on_begin()

        with torch.no_grad(), self.test_timer:
            for idx, (image, target) in tqdm(enumerate(self.loader_test)):
                output = self.model(image)
                target = (target * 255).clamp(0, 255).round()
                output = (output * 255).clamp(0, 255).round()
                if self.args.test_show:
                    pos = random.randint(0, len(target) - 1)
                    for name, tensor in {'target': target, 'corrected': output}.items():
                        image = tensor[pos].to(torch.uint8).squeeze().permute(1, 2, 0).cpu().numpy()
                        if self.args.color_space.inverse_code:
                            image = cv2.cvtColor(image, self.args.color_space.inverse_code)
                        cv2.imshow(name, image)
                    cv2.waitKey(1)

                for metric in self.similarity:
                    quality[metric.name].append(metric(output[:, :, 3:-3, 3:-3], target[:, :, 3:-3, 3:-3]).item())

            for key in quality.keys():
                value = np.mean(quality[key])
                if value > self.eval_stats[key][0]:
                    self.eval_stats[key] = value, self.epoch

            logs = {
                'current': {item: np.mean(quality[item])for item in quality.keys()},
                'best': self.eval_stats,
                'elapsed_time': self.test_timer.mean
            }
            self.test_timer.reset()
            for metric_name in quality.keys():
                Logger.current_logger().report_scalar("test", metric_name, iteration=self.epoch,
                                                      value=np.mean(quality[metric_name]))
        self.eval_callback.on_end(self.epoch, logs)

        if not self.args.test_only:
            self.freeze_model()

    def terminate(self) -> bool:
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs


def parse_opt() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--color_space', type=ColorSpace, default='hsv',
                        help='Colorspace type of input images.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Size of sliding window.')

    parser.add_argument('--train_data', type=str, nargs='+', default=[f'{root}/LOL-v2/Synthetic/Train'],
                        help='Path to train dataset images.')
    parser.add_argument('--test_data', type=str, nargs='+', default=[f'{root}/LOL-v2/Synthetic/Test'],
                        help='Path to test dataset images.')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--device', default='cuda:0', help='cuda:0, ... or cpu.')
    parser.add_argument('--skip_threshold', type=float, default=1e2,
                        help='Skipping batch that has large error.')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
    parser.add_argument('--test_only', action='store_true', help='Set this option to test the model.')
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

    parser.add_argument('--load_path', type=str, default='', help='Path to weights for loading.')
    parser.add_argument('--test_show', type=bool, default=True, help='Show visual correction results.')
    parser.add_argument('--loss', type=LossType, default='combine', help='Choose loss type.')
    options = parser.parse_args()
    return options


if __name__ == '__main__':
    from datetime import datetime
    from warnings import filterwarnings

    random.seed(15236)
    init()
    filterwarnings('ignore')
    clock = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    print(f'{Fore.LIGHTGREEN_EX}Start experiment at: {clock}{Style.RESET_ALL}')
    root = Path(__file__).parent

    args = parse_opt()
    print(f'{Fore.LIGHTGREEN_EX}Color space: {args.color_space.value}, kernel size: {args.kernel_size}{Style.RESET_ALL}')

    task = Task.init(project_name='Light correction',
                     task_name=f'color space:{args.color_space}; kernel-size:{args.kernel_size}')
    worker = Trainer(args)
    try:
        while not worker.terminate():
            worker.train()
            worker.test()
    except StopIteration:
        print(traceback.format_exc())
    task.close()
    print(f'{Fore.LIGHTGREEN_EX}Train finish!{Style.RESET_ALL}')
