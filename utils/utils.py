import torch.nn
import operator
from typing import Dict, List
from utils.pytorch_metrics import GMSD, PSNR, LPIPS, SSIM


class EarlyStopping(object):
    def __init__(self, patience: int = 10, min_delta: float = 0.0, metrics: List[str] = [PSNR.__name__]) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.metrics = metrics
        self.best_scores = {metric: None for metric in metrics}
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_scores: Dict[str, float]) -> bool:
        improvement_detected = False

        for metric in self.metrics:
            if self.best_scores[metric] is None:
                self.best_scores[metric] = current_scores[metric]
                improvement_detected = True
            else:
                if self.check_improvement(metric, current_scores[metric]):
                    improvement_detected = True

        if not improvement_detected:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

        return self.early_stop

    def check_improvement(self, metric: str, current_value: float) -> bool:
        if metric in [PSNR.__name__, SSIM.__name__]:
            comparison_operator = operator.gt
            threshold = self.best_scores[metric] + self.min_delta
        elif metric in [GMSD.__name__, LPIPS.__name__]:
            comparison_operator = operator.lt
            threshold = self.best_scores[metric] - self.min_delta
        else:
            raise NotImplementedError(f"Metric {metric} is not support.")

        if comparison_operator(current_value, threshold):
            self.best_scores[metric] = current_value
            return True
        return False


def get_flops(model: torch.nn.Module, tensor: torch.Tensor) -> None:
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, tensor)
    print(f"Total FLOPs: {flops.total() / 1e9} GFLOPs")
