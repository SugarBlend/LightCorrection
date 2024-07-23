import abc
import logging
from time import perf_counter
from typing import Any, Dict, Optional, List
from colorama import Fore, Style
from abc import ABCMeta
import json


class Logger(metaclass=ABCMeta):
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.epoch_start_time: float = 0
        self.training_logs: List[Dict[str, Any]] = []

    @abc.abstractmethod
    def on_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abc.abstractmethod
    def on_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass


class TrainingLogger(Logger):
    def __init__(self, log_interval: int = 10):
        super().__init__(log_interval)
        self.training_logs: List[Dict[str, Any]] = []

    def on_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.epoch_start_time = perf_counter()
        logging.info(f"{Fore.LIGHTBLUE_EX}Epoch {logs['epoch'] + 1} starting, "
                     f"learning rate: {logs['lr']:.4f}{Style.RESET_ALL}")

    def on_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        elapsed_time = perf_counter() - self.epoch_start_time
        logging.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logs['epoch_time'] = elapsed_time
        self.training_logs.append(logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if (batch + 1) % self.log_interval == 0:
            logging.info(f"Batch {batch + 1}: Loss = {logs['loss']:.4f}, Batch time = {logs['batch time']:.4f}s.")


class TestLogger(Logger):
    def __init__(self, log_interval: int = 10):
        super().__init__(log_interval)
        self.training_logs: List[Dict[str, Any]] = []

    def on_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.epoch_start_time = perf_counter()
        logging.info(f"{Fore.LIGHTBLUE_EX}Evaluation step:{Style.RESET_ALL}")

    def on_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        elapsed_time = perf_counter() - self.epoch_start_time
        logging.info(f"Evaluation {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logging.info(json.dumps(logs))
        logs['epoch_time'] = elapsed_time
        self.training_logs.append(logs)


class EarlyStopping(object):
    def __init__(self, patience: int = 3):
        self.patience: int = patience
        self.counter: int = 0
        self.best_loss: Optional[float] = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        current_loss = logs['loss']
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logging.info("Early stopping triggered.")
                return True
        return False
