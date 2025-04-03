import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int) -> None:
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.Mish(),
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.Mish()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MLPDenoiser(nn.Module):
    def __init__(self, kernel_size: int, num_of_layers: int, channels: int = 3) -> None:
        super(MLPDenoiser, self).__init__()
        self.kernel_size = kernel_size
        self.num_of_layers = num_of_layers
        self.channels = channels

        self.extract_patches = nn.Conv2d(
            in_channels=channels,
            out_channels=channels * (kernel_size ** 2),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        )

        layers = [ResidualBlock(kernel_size**2) for _ in range(num_of_layers)]
        layers.append(nn.Linear(kernel_size**2, 1))

        self.layers = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        patches = self.extract_patches(x)
        patches = patches.view(b, c, int(self.kernel_size ** 2), int(h * w))
        patches = patches.permute(0, 1, 3, 2)

        out = self.layers(patches)
        out = out.reshape(b, c, h, w)

        return x - out


if __name__ == '__main__':
    channels = 3
    model = MLPDenoiser(9, 5, channels)
    x = torch.ones(1, channels, 400, 600)
    out = model(x)
    assert x.shape == out.shape
    torch.jit.script(model)
