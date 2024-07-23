import numpy as np
from contextlib import ContextDecorator
from collections import deque
from time import perf_counter


class Timer(ContextDecorator):
    def __init__(self):
        self.elapsed_time = deque(maxlen=60)

    def __enter__(self):
        self.start_time = perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time.append(perf_counter() - self.start_time)

    @property
    def mean(self) -> float:
        return np.mean(self.elapsed_time)

    @property
    def current(self) -> float:
        return self.elapsed_time[-1]

    def reset(self) -> None:
        self.elapsed_time.clear()
