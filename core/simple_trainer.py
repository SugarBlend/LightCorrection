import torch
import mlflow
import clearml
import logging
from tqdm import tqdm
from argparse import Namespace
from torch.cuda.amp import autocast
from typing import Callable, Optional, Union, Dict
from core.base_trainer import BaseTrainer
from utils.union_dataloader import DataLoader


class SimpleTrainer(BaseTrainer):
    def __init__(
            self,
            args: Namespace,
            active_logger: Optional[Union[clearml.Task, mlflow.ActiveRun]],
            logger: logging.Logger,
            model: Dict[str, torch.nn.Module],
            optimizer: Dict[str, torch.optim.Optimizer],
            scheduler: Dict[str, torch.optim.lr_scheduler.LRScheduler],
            criterion: Dict[str, Callable],
            loader_train: DataLoader,
            loader_test: DataLoader
    ) -> None:
        super(SimpleTrainer, self).__init__(args, active_logger, logger, loader_train, loader_test)
        self.args = args
        assert len(model) == 1
        for model_name, torch_model in model.items():
            self.model_name, self.model = model_name, torch_model
        self.model.to(self.device)
        self.criterion: Callable = criterion[self.model_name].to(self.device)
        self.optimizer: torch.optim.Optimizer = optimizer[self.model_name]
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = scheduler[self.model_name]

    def train(self) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.loader_train, dynamic_ncols=True, leave=False,
                                                           ncols=100, desc='Train iteration', mininterval=5)):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            if self.args.datasets["normalization_range"] == [-1, 1]:
                inputs, targets = [item * 2 - 1 for item in [inputs, targets]]

            with autocast(enabled=self.args.training["mixed_precision"]):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            running_loss += loss.item()

        self.scheduler.step()
        avg_loss = running_loss / len(self.loader_train)
        if self.args.logging["backend"] == "mlflow":
            mlflow.log_metric(key="Train Loss", value=avg_loss, step=self.current_epoch)
        elif self.args.logging["backend"] == "clearml":
            self.logger.report_scalar("Loss", "Train", value=avg_loss, iteration=self.current_epoch)

        torch.cuda.empty_cache()
        return {self.model_name: avg_loss}
