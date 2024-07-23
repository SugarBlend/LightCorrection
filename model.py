import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, kernel_size: int):
        super(Model, self).__init__()
        self.kernel_size = kernel_size

        self._modules['Input'] = self.define_hidden_block(self.kernel_size**2, 2 * self.kernel_size**2)

        self._modules['Hidden_1'] = self.define_hidden_block(2 * self.kernel_size**2, 3 * self.kernel_size**2)

        self._modules['Hidden_2'] = self.define_hidden_block(3 * self.kernel_size**2, 3 * self.kernel_size**2)

        self._modules['Hidden_3'] = self.define_hidden_block(3 * self.kernel_size**2, 2 * self.kernel_size**2)

        self._modules['Output'] = self.define_hidden_block(2 * self.kernel_size**2, 1)

    def define_hidden_block(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=out_features),
            nn.BatchNorm3d(num_features=3),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        padded_x = F.pad(x, pad=[self.kernel_size // 2] * 4, mode='reflect')
        sliding_windows = padded_x.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        x = sliding_windows.reshape(b, c, h, w, -1)

        for block in self._modules.values():
            x = block(x)

        return x.reshape(b, c, h, w)


if __name__ == '__main__':
    model = Model(3)
    x = torch.arange(12*3*600*400).float().reshape(12, 3, 600, 400)
    model(x)
