import torch
import mlflow
import logging
import clearml
import argparse
from tqdm import tqdm
from core.base_trainer import BaseTrainer
from utils.union_dataloader import DataLoader
from typing import Dict, Tuple, List, Any, Optional, Callable, Union
from utils.losses import ContentLoss, PixelLoss


class GANTrainer(BaseTrainer):
    def __init__(
            self,
            args: argparse.Namespace,
            active_logger: Optional[Union[clearml.Task, mlflow.ActiveRun]],
            logger: logging.Logger,
            models: Dict[str, torch.nn.Module],
            optimizers: Dict[str, torch.optim.Optimizer],
            schedulers: Dict[str, torch.optim.lr_scheduler.LRScheduler],
            losses: Dict[str, torch.nn.Module],
            loader_train: DataLoader,
            loader_test: DataLoader
    ) -> None:
        super(GANTrainer, self).__init__(args, active_logger, logger, loader_train, loader_test)

        for variable in [models, optimizers, schedulers, losses]:
            assert all(model in variable for model in ['discriminator', 'generator'])

        for model in models:
            self.__dict__[f'{model}']: torch.nn.Module = models[model].to(self.device)
            self.__dict__[f'{model}_loss']: torch.nn.Module = losses[model].to(self.device)

            self.__dict__[f'{model}_optimizer']: torch.optim.Optimizer = optimizers[model]
            self.__dict__[f'{model}_scheduler']: torch.optim.lr_scheduler.LRScheduler = schedulers[model]

        self.model = self.generator.to(self.device)
        self.pixel_weight: float = 1.0
        self.pixel_criterion = torch.nn.MSELoss().to(self.device)
        # self.pixel_criterion = PixelLoss().to(self.device)
        self.feature_weight: float = 1.0
        self.content_criterion = ContentLoss(
            "vgg19", False, 1000, "", ["features.35"],
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ).to(self.device)
        self.adversarial_weight: float = 0.001
        self.adversarial_criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

    def train(self) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        d_running_loss, g_running_loss = 0.0, 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(self.loader_train, dynamic_ncols=True, leave=False,
                                                           ncols=100, desc='Train iteration', mininterval=5)):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            if self.args.datasets["normalization_range"] == [-1, 1]:
                inputs, targets = [item * 2 - 1 for item in [inputs, targets]]

            batch_size = inputs.shape[0]
            real_label = torch.full([batch_size, 1], 1.0, dtype=torch.float, device=self.device)
            fake_label = torch.full([batch_size, 1], 0.0, dtype=torch.float, device=self.device)

            for d_parameters in self.discriminator.parameters():
                d_parameters.requires_grad = False
            self.generator.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.args.training["mixed_precision"]):
                generated_images = self.generator(targets)
                self.discriminator_optimizer.zero_grad(set_to_none=True)

                pixel_loss = self.pixel_criterion(generated_images, targets)
                feature_loss = self.content_criterion(generated_images, targets)
                adversarial_loss = self.adversarial_criterion(self.discriminator(generated_images), real_label)
                pixel_loss = torch.sum(torch.mul(self.pixel_weight, pixel_loss))
                feature_loss = torch.sum(torch.mul(self.feature_weight, feature_loss))
                adversarial_loss = torch.sum(torch.mul(self.adversarial_weight, adversarial_loss))
                # Compute generator total loss
                generator_loss = pixel_loss + feature_loss + adversarial_loss

            self.scaler.scale(generator_loss).backward()
            self.scaler.step(self.generator_optimizer)
            self.scaler.update()
            # end training generator model

            # start training the discriminator model
            # During discriminator model training, enable discriminator model backpropagation
            for d_parameters in self.discriminator.parameters():
                d_parameters.requires_grad = True
            self.discriminator.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.args.training["mixed_precision"]):
                target_prediction = self.discriminator(targets)
                discriminator_loss_gt = self.adversarial_criterion(target_prediction, real_label)
            self.scaler.scale(discriminator_loss_gt).backward()

            # Calculate the classification score of the generated samples by the discriminator model
            with torch.cuda.amp.autocast(enabled=self.args.training["mixed_precision"]):
                fake_prediction = self.discriminator(generated_images.detach().clone())
                discriminator_loss_generated = self.adversarial_criterion(fake_prediction, fake_label)
            self.scaler.scale(discriminator_loss_generated).backward()

            # Compute the discriminator total loss value
            discriminator_loss = discriminator_loss_gt + discriminator_loss_generated
            self.scaler.step(self.discriminator_optimizer)
            self.scaler.update()

            g_running_loss += generator_loss.item()
            d_running_loss += discriminator_loss.item()

        for scheduler in [self.generator_scheduler, self.discriminator_scheduler]:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(generator_loss)
            else:
                scheduler.step()

        for name, loss in zip(["Generator", "Discriminator"], [g_running_loss, d_running_loss]):
            avg_loss = loss / len(self.loader_train)
            if self.args.logging["backend"] == "clearml":
                self.logger.report_scalar(f"{name} Loss", "Train", value=avg_loss, iteration=self.current_epoch)
            elif self.args.logging["backend"] == "mlflow":
                mlflow.log_metric(key=f"{name} Loss", value=avg_loss, step=self.current_epoch)
        torch.cuda.empty_cache()

        return {"Generator loss": g_running_loss / len(self.loader_train),
                "Discriminator loss": d_running_loss / len(self.loader_train)}
