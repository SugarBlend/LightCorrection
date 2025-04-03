import cv2
import torch
import mlflow
import clearml
import logging
import tempfile
import numpy as np
from tqdm import tqdm, trange
from argparse import Namespace
from torch.cuda.amp import GradScaler
from abc import abstractmethod, ABCMeta
from typing import Dict, Optional, Union

from utils.utils import EarlyStopping
from utils.union_dataloader import DataLoader
from utils.pytorch_metrics import PSNR, SSIM, GMSD, LPIPS
from utils.visualization import MatplotlibDrawer, OpenCVDrawer
from utils.parser import dynamic_import


class BaseTrainer(metaclass=ABCMeta):
    def __init__(
            self,
            args: Namespace,
            active_logger: Optional[Union[clearml.Task, mlflow.ActiveRun]],
            logger: logging.Logger,
            loader_train: DataLoader,
            loader_test: DataLoader
    ) -> None:
        self.args = args
        self.active_logger: Optional[clearml.Task] = active_logger.get_logger() if (
            isinstance(active_logger, clearml.Task)) else None
        self.logger: logging.Logger = logger
        self.loader_train: DataLoader = loader_train
        self.loader_test: DataLoader = loader_test

        self.device = torch.device(self.args.training["device"])
        self.scaler = GradScaler(enabled=self.args.training["mixed_precision"])
        self.early_stopping = EarlyStopping(**self.args.training["early_stopping"],
                                            metrics=self.args.training["validation"]["metrics"])
        if self.args.logging["backend"] == "clearml":
            self.task = active_logger
            self.task.connect(vars(args))

        metrics = [dynamic_import(self.args.training["validation"]["module"], metric_name)
                   for metric_name in self.args.training["validation"]["metrics"]]

        self.quality = {metric.__name__: metric() for metric in metrics}
        self.drawer = OpenCVDrawer()
        self.current_epoch: Optional[int] = None

        self.validation_loss = torch.nn.MSELoss().to(self.device)

    @abstractmethod
    def train(self) -> Union[float, str]:
        pass

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        self.model.eval()
        total_qualities = {metric: 0.0 for metric in self.quality}
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(self.loader_test, dynamic_ncols=True, leave=False, ncols=100,
                                                           desc='Validation', mininterval=5)):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            if self.args.datasets["normalization_range"] == [-1, 1]:
                inputs = inputs * 2 - 1

            outputs = self.model(inputs)

            if self.args.datasets["normalization_range"] == [-1, 1]:
                inputs, outputs = [((item + 1) * 128).clip(0, 255) for item in [inputs, outputs]]
                targets = targets.clip(0, 1) * 255
            else:
                inputs, outputs, targets = [item.clip(0, 1) * 255 for item in [inputs, outputs, targets]]
            running_loss += self.validation_loss(outputs, targets).item()

            if self.args.training["show_test"]:
                self.drawer.visualize_images(inputs, outputs, targets)

            if batch_idx == 0:
                correspondence = {"Input": inputs, "Output": outputs, "Ground Truth": targets}
                for series, tensors in correspondence.items():
                    image = tensors[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                    if self.args.logging["backend"] == "clearml":
                        self.active_logger.report_image("Results", series, iteration=self.current_epoch, image=image)
                    elif self.args.logging["backend"] == "mlflow":
                        mlflow.log_image(image=image, artifact_file=f"test/{self.current_epoch}/{series}.png")
                        # mlflow.log_image(image=image, key=series, step=self.current_epoch)

                mae_map = torch.abs(outputs - targets).mean(dim=0).permute(1, 2, 0).cpu().numpy()
                mae_map = (mae_map / mae_map.max() * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(mae_map, cv2.COLORMAP_JET)

                if self.args.logging["backend"] == "clearml":
                    self.active_logger.report_image("Error Heatmap", "MAE Map", iteration=self.current_epoch,
                                                    image=heatmap)
                elif self.args.logging["backend"] == "mlflow":
                    mlflow.log_image(image=heatmap, artifact_file=f"test/{self.current_epoch}/MAE.png")
                    # mlflow.log_image(image=heatmap, key="MAE Map", step=self.current_epoch)

            for metric in total_qualities:
                total_qualities[metric] += self.quality[metric](outputs, targets)

        avg_loss = running_loss / len(self.loader_test)
        if self.args.logging["backend"] == "mlflow":
            mlflow.log_metric(key="Test Loss", value=avg_loss, step=self.current_epoch)
        elif self.args.logging["backend"] == "clearml":
            self.logger.report_scalar("Loss", "Test", value=avg_loss, iteration=self.current_epoch)

        for metric in self.quality:
            total_qualities[metric] /= len(self.loader_test)
            if self.args.logging["backend"] == "mlflow":
                mlflow.log_metric(metric, total_qualities[metric], step=self.current_epoch)
            elif self.args.logging["backend"] == "clearml":
                self.active_logger.report_scalar(metric, "Validation", value=total_qualities[metric],
                                                 iteration=self.current_epoch)

        torch.cuda.empty_cache()
        return total_qualities

    def run(self) -> str:
        best_qualities = None
        self.current_epoch: int = 0
        best_model_path: str = ''

        for epoch in trange(self.args.training["epochs"], dynamic_ncols=True, leave=False, ncols=100,
                            desc='Epoch progress'):
            train_loss = self.train()
            for model_name, loss in train_loss.items():
                report_message = f'Error for {model_name} on {epoch} epoch: {loss}'
                if self.args.logging["backend"] == "clearml":
                    self.active_logger.report_text(report_message)
                else:
                    self.logger.info(report_message)

            total_qualities = self.test()
            if self.is_improvement(best_qualities, total_qualities):
                best_qualities = total_qualities
                with tempfile.TemporaryDirectory() as tempdir:
                    best_model_path = f'{tempdir}/best_model_epoch_{epoch}.pth'
                    torch.save({'model_state_dict': self.model.state_dict(), 'args': vars(self.args)},
                               best_model_path)
                    report_message = f"New best model saved with: " + ", ".join(
                        f"{metric}: {total_qualities[metric]:.3f}" for metric in total_qualities
                    )

                    if self.args.logging["backend"] == "mlflow":
                        for artifact_path, model in {
                            f"checkpoints/best_model_epoch_{epoch}": self.model,
                            # f"checkpoints/scripted_model_epoch_{epoch}": torch.jit.script(self.model)
                        }.items():
                            mlflow.pytorch.log_model(pytorch_model=model, artifact_path=artifact_path)
                        self.logger.info(report_message)
                    elif self.args.logging["backend"] == "clearml":
                        self.task.upload_artifact(name=self.model.__class__.__name__, artifact_object=best_model_path,
                                                  wait_on_upload=True)
                        self.active_logger.report_text(report_message)

            if self.early_stopping(total_qualities):
                break

            self.current_epoch += 1
            torch.cuda.empty_cache()

        return best_model_path

    @staticmethod
    def is_improvement(best_qualities: Dict[str, float], total_qualities: Dict[str, float]) -> bool:
        if best_qualities is None:
            return True

        improvements = 0
        for metric in best_qualities:
            if metric in [PSNR.__name__, SSIM.__name__]:
                improvements += int(total_qualities[metric] > best_qualities[metric])
            elif metric in [GMSD.__name__, LPIPS.__name__]:
                improvements += int(total_qualities[metric] < best_qualities[metric])
            else:
                raise NotImplementedError

        return improvements >= 3
