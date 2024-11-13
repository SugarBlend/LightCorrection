import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.register_module('pad', torch.nn.ReflectionPad2d(self.kernel_size // 2))
        self.register_module('unfold', torch.nn.Unfold(self.kernel_size))
        self.register_module('input', self.define_hidden_block(3 * self.kernel_size**2, 2 * self.kernel_size**2))
        self.register_module('hidden_1', self.define_hidden_block(2 * self.kernel_size**2, 3 * self.kernel_size**2))
        self.register_module('hidden_2', self.define_hidden_block(3 * self.kernel_size**2, 3 * self.kernel_size**2))
        self.register_module('hidden_3', self.define_hidden_block(2 * self.kernel_size**2, 2 * self.kernel_size**2))
        self.register_module('hidden_4', self.define_hidden_block(2 * self.kernel_size**2, 5 * self.kernel_size**2))
        self.register_module('hidden_5', self.define_hidden_block(5 * self.kernel_size**2, 2 * self.kernel_size**2))
        self.register_module('output', self.define_hidden_block(2 * self.kernel_size**2, 3))
        self.register_module('enc', self.encoder(3 * self.kernel_size**2))
        self.register_module('enc_2', self.encoder(2 * self.kernel_size**2))
        self.register_module('dec', self.decoder(2 * self.kernel_size**2))

    def define_hidden_block(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
        )

    def encoder(self, in_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features*3),
            nn.Linear(in_features=in_features*3, out_features=in_features),
            nn.Linear(in_features=in_features, out_features=3),
        )

    def decoder(self, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features=3, out_features=out_features),
            nn.Linear(in_features=out_features, out_features=out_features*3),
            nn.Linear(in_features=out_features*3, out_features=out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        padded_x = F.pad(x, pad=[self.kernel_size // 2] * 4, mode='reflect')
        sliding_windows = self.unfold(padded_x).permute(0, 2, 1)

        x1 = self.input(sliding_windows)
        x2 = F.leaky_relu(self.hidden_1(x1))
        x3 = F.tanh(self.hidden_2(x2))
        x3_enc = self.enc(x3)
        x4 = F.leaky_relu(x3_enc + self.enc_2(x1))
        x4_dec = self.hidden_3(self.dec(x4))
        x5 = F.tanh(self.hidden_4(x4_dec))
        x6 = self.hidden_5(x5) - x1
        x7 = F.leaky_relu(self.output(x6)).permute(0, 2, 1).reshape(b, c, h, w)

        return x7


if __name__ == '__main__':
    model = Model(3)
    x = torch.arange(8*3*600*400).float().reshape(8, 3, 400, 600)
    model(x)

    scripted_model = torch.jit.script(model)
    scripted_model.save('./scripted_model.pt')
