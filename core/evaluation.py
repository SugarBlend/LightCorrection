import os
import torch
import random
import clearml
import mlflow
import pathlib
import warnings
import argparse
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Tuple
from utils.parser import dynamic_import
from utils.union_dataloader import UnionDataset, DataLoader, default_collate


class Evaluator(object):
    def __init__(
            self,
            args: argparse.Namespace,
            test_dataset: Dataset,
            task: Optional[clearml.Task] = None
    ) -> None:
        self.args = args
        self.test_dataset: Dataset = test_dataset
        self.task: Optional[clearml.Task] = task

        metrics = [dynamic_import(self.args.training["validation"]["module"], metric_name)
                   for metric_name in self.args.training["validation"]["metrics"]]
        self.metrics = {metric.__name__: metric() for metric in metrics}

    @torch.no_grad()
    def validation(
            self,
            model: torch.nn.Module, loader_test: DataLoader
    ) -> Tuple[Dict[str, float], List[Tuple[int, List[np.ndarray]]]]:
        images: List[Tuple[int, List[np.ndarray]]] = []
        total_quality = {metric: [] for metric in self.metrics}
        random.seed(self.args.training["seed"])
        samples = random.sample(range(len(loader_test)), k=2)

        for batch_idx, (inputs, targets) in enumerate(tqdm(loader_test, desc='Evaluation step')):
            inputs, targets = (inputs.to(self.args.training["device"], non_blocking=True),
                               targets.to(self.args.training["device"], non_blocking=True))
            if self.args.datasets["normalization_range"] == [-1, 1]:
                inputs = inputs * 2 - 1

            outputs = model(inputs)

            if self.args.datasets["normalization_range"] == [-1, 1]:
                inputs, outputs = [((item + 1) * 128).clip(0, 255) for item in [inputs, outputs]]
                targets = targets.clip(0, 1) * 255
            else:
                inputs, outputs, targets = [item.clip(0, 1) * 255 for item in [inputs, outputs, targets]]

            if self.args.training["validation"]["collect_outputs"] and batch_idx in samples:
                images.append((batch_idx, [item.to(torch.uint8).cpu().numpy() for item in [inputs, outputs, targets]]))

            if self.args.training["validation"]["calculate_metrics"]:
                for metric in self.metrics:
                    total_quality[metric].append(self.metrics[metric](outputs.float(), targets.float()))

        if self.args.training["validation"]["calculate_metrics"]:
            total_quality = {metric: np.mean(list(filter(lambda x: x < np.inf, total)))
                             for metric, total in total_quality.items()}

        return total_quality, images

    def evaluate(self, model: torch.nn.Module, std_range: Optional[range] = None) -> None:
        model.eval()
        noise_levels = std_range if isinstance(std_range, List) else [None]
        simulation_results = {}

        for sigma in noise_levels:
            inputs_transform = A.Compose([A.GaussNoise(std_range=(sigma / 255, sigma / 255), p=1.0)]) if sigma else None
            self.test_dataset.inputs_transform = inputs_transform

            loader_test = DataLoader(self.test_dataset, batch_size=1, shuffle=False, collate_fn=default_collate)
            scores, images = self.validation(model, loader_test)
            simulation_results[sigma] = {'metrics': scores, 'collages': images}

        if self.args.training["validation"]["calculate_metrics"]:
            self.save_metrics_to_excel(simulation_results)

        if self.args.training["validation"]["collect_outputs"]:
            self.save_collages(simulation_results)

    def save_metrics_to_excel(
            self,
            simulation_results: Dict[str, Dict[str, float]]
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            df_list = [pd.DataFrame(data["metrics"], index=[sigma]).rename_axis("Sigma")
                       for sigma, data in simulation_results.items()]
            merged_df = pd.concat(df_list)
            statistic_file = f"{temp_dir}/validation_metrics.xlsx"
            merged_df.to_excel(statistic_file)

            if self.args.logging["backend"] == "clearml":
                self.task.upload_artifact(name='metrics_report', artifact_object=statistic_file, wait_on_upload=True)

                self.task.get_logger().report_table(
                    title="Evaluation Metrics",
                    series="Metrics Table",
                    iteration=0,
                    table_plot=merged_df
                )
            elif self.args.logging["backend"] == "mlflow":
                mlflow.log_table(pd.DataFrame(merged_df.index, index=merged_df.index).join(merged_df),
                                 artifact_file="evaluation/validation_metrics.json")
                mlflow.log_artifact(local_path=statistic_file, artifact_path="evaluation")

    def save_collages(self, simulation_results: Dict[str, Dict[str, Tuple[int, List[np.ndarray]]]]) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for sigma, data in simulation_results.items():
                for pos, (input_img, output_img, target_img) in data["collages"]:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    titles = ["Input", "Output", "Target"]

                    for ax, img, title in zip(axes, [input_img, output_img, target_img], titles):
                        ax.imshow(img[0].transpose(1, 2, 0).astype(np.uint8))
                        ax.set_title(title)
                        ax.axis("off")

                    filename = Path(self.test_dataset.inputs_paths[pos]).stem
                    if self.args.logging["backend"] == "clearml":
                        self.task.get_logger().report_matplotlib_figure(
                            title="Evaluation Collages",
                            series=f"Image - {filename}, AWGN σ = {sigma}" if sigma else f"Image - {filename}",
                            iteration=0,
                            figure=fig
                        )
                    elif self.args.logging["backend"] == "mlflow":
                        filename = f"Image-{filename}_σ_{sigma}" if sigma else f"Image-{filename}"
                        save_path = f"{temp_dir}/{filename}.png"
                        fig.savefig(save_path, dpi=300)
                        mlflow.log_figure(fig, f"evaluation/collages/{filename}.png")

                    plt.close(fig)


if __name__ == '__main__':
    root = pathlib.Path(os.getcwd()).parent.parent

    logging_backend = "mlflow"  # "clearml"
    task_name = "Image Quality Evaluation"
    project_name = "Image Quality Evaluation"

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from models.models.kan_denoiser import KANDenoiser

        active_logger = None
        if logging_backend == "mlflow":
            mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
            mlflow.set_experiment(project_name)
            mlflow.start_run(run_name=task_name)
            active_logger = mlflow.active_run()
        elif logging_backend == "clearml":
            task = clearml.Task.init(task_name="Image Quality Evaluation", project_name="Image Quality Evaluation")
            active_logger = task

        model = KANDenoiser(kernel_size=5).to('cuda:0')

        dataset = UnionDataset(
            data={
                "low": [f"{root}/LOL-v2/Synthetic/Test/Normal"],
                "target": [f"{root}/LOL-v2/Synthetic/Test/Normal"],
            }
        )

        args = argparse.Namespace(
            **dict(
                training=dict(
                    device="cuda:0",
                    seed=42,
                    validation=dict(
                        module='models.utils.pytorch_metrics',
                        metrics=['PSNR', 'SSIM', 'GMSD', 'LPIPS'],
                        calculate_metrics=True,
                        collect_outputs=True
                    )
                ),
                datasets=dict(
                    normalization_range=[-1, 1]
                ),
                logging=dict(
                    backend=logging_backend
                )
            )
        )
        evaluator = Evaluator(args, dataset, active_logger)
        evaluator.evaluate(model, range(5, 26, 5))

        if logging_backend == "mlflow":
            mlflow.end_run()
        elif logging_backend == "clearml":
            task.close()
