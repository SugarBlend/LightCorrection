import os
import torch
from torch import nn, Tensor
from pytorch_ssim import SSIM
from torch.nn import functional as F_torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Any, cast, Dict, List, Union


feature_extractor_net_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(net_cfg_name: str, batch_norm: bool = False) -> nn.Sequential:
    net_cfg = feature_extractor_net_cfgs[net_cfg_name]
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in net_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v

    return layers


class _FeatureExtractor(nn.Module):
    def __init__(
            self,
            net_cfg_name: str = "vgg19",
            batch_norm: bool = False,
            num_classes: int = 1000) -> None:
        super(_FeatureExtractor, self).__init__()
        self.features = _make_layers(net_cfg_name, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            net_cfg_name: str,
            batch_norm: bool,
            num_classes: int,
            model_weights_path: str,
            feature_nodes: list,
            feature_normalize_mean: list,
            feature_normalize_std: list,
    ) -> None:
        super(ContentLoss, self).__init__()
        # Define the feature extraction model
        model = _FeatureExtractor(net_cfg_name, batch_norm, num_classes)
        # Load the pre-trained model
        if model_weights_path == "":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif model_weights_path is not None and os.path.exists(model_weights_path):
            checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
            if "state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError("Model weight file not found")
        # Extract the output of the feature extraction layer
        self.feature_extractor = create_feature_extractor(model, feature_nodes)
        # Select the specified layers as the feature extraction layer
        self.feature_extractor_nodes = feature_nodes
        # input normalization
        self.normalize = transforms.Normalize(feature_normalize_mean, feature_normalize_std)
        # Freeze model parameters without derivatives
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, generated: Tensor, target: Tensor) -> [Tensor]:
        assert generated.size() == target.size(), "Two tensor must have the same size"
        device = generated.device

        losses = []
        # input normalization
        sr_tensor = self.normalize(generated)
        gt_tensor = self.normalize(target)

        # Get the output of the feature extraction layer
        sr_feature = self.feature_extractor(sr_tensor)
        gt_feature = self.feature_extractor(gt_tensor)

        # Compute feature loss
        for i in range(len(self.feature_extractor_nodes)):
            losses.append(F_torch.mse_loss(sr_feature[self.feature_extractor_nodes[i]],
                                           gt_feature[self.feature_extractor_nodes[i]]))

        losses = torch.Tensor([losses]).to(device)

        return losses


class PerceptualLoss(nn.Module):
    def __init__(self, layers: List[int] = [2, 7, 12], lambda_p: float = 0.05) -> None:
        super(PerceptualLoss, self).__init__()
        import torchvision.models as models
        self.vgg = models.vgg19(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.layers = layers
        self.criterion = nn.L1Loss()
        self.lambda_p = lambda_p

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = torch.zeros((1, 1))
        for i, layer in enumerate(self.vgg):
            generated = layer(generated)
            target = layer(target)
            if i in self.layers:
                loss += self.criterion(generated, target)
        return self.lambda_p * loss


class PixelLoss(torch.nn.Module):
    def __init__(self):
        super(PixelLoss, self).__init__()
        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_ssim = SSIM()

    def forward(
            self,
            generated: torch.Tensor,
            target: torch.Tensor,
            structural_weight: float = 0.2,
            pixel_weight: float = 0.8
    ) -> torch.Tensor:
        return (pixel_weight * self.criterion_l1(generated, target) +
                structural_weight * (1 - self.criterion_ssim(generated, target)))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight: float = 1.0) -> None:
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, generated: torch.Tensor) -> torch.Tensor:
        batch_size = generated.size()[0]
        h_x = generated.size()[2]
        w_x = generated.size()[3]
        count_h = self.tensor_size(generated[:, :, 1:, :])
        count_w = self.tensor_size(generated[:, :, :, 1:])
        h_tv = torch.pow((generated[:, :, 1:, :] - generated[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((generated[:, :, :, 1:] - generated[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t: torch.Tensor) -> int:
        return t.size()[1] * t.size()[2] * t.size()[3]


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    @staticmethod
    def forward(real_predictions: torch.Tensor, fake_predictions: torch.Tensor) -> torch.Tensor:
        real_loss = torch.mean(torch.relu(1.0 - real_predictions))
        fake_loss = torch.mean(torch.relu(1.0 + fake_predictions))
        return real_loss + fake_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0) -> None:
        super(DiscriminatorLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing

    def forward(self, real_predictions: torch.Tensor, fake_predictions: torch.Tensor) -> torch.Tensor:
        real_labels = torch.full_like(real_predictions, 1.0 - self.label_smoothing)
        fake_labels = torch.full_like(fake_predictions, self.label_smoothing)

        real_loss = self.criterion(real_predictions, real_labels)
        fake_loss = self.criterion(fake_predictions, fake_labels)

        return (real_loss + fake_loss) / 2


class GeneratorLossV1(nn.Module):
    def __init__(
            self,
            perceptual_weight: float = 0.1,
            adv_weight: float = 0.01,
            pixel_weight: float = 1.0,
            pixel_loss: str = "l2"
    ) -> None:
        super(GeneratorLossV1, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.adv_weight = adv_weight
        self.pixel_weight = pixel_weight

        self.perceptual_loss = PerceptualLoss()
        if pixel_loss == "l2":
            self.pixel_loss = nn.MSELoss()
        else:
            self.pixel_loss = nn.L1Loss()
        self.adv_loss = nn.BCEWithLogitsLoss()

    def forward(self, generated: torch.Tensor, target: torch.Tensor, fake_predictions: torch.Tensor) -> torch.Tensor:
        pixel_loss = self.pixel_loss(generated, target)
        perceptual_loss = self.perceptual_loss(generated, target)
        adv_loss = self.adv_loss(fake_predictions, torch.ones_like(fake_predictions))

        total_loss = (
            self.pixel_weight * pixel_loss +
            self.perceptual_weight * perceptual_loss +
            self.adv_weight * adv_loss
        )
        return total_loss


class GeneratorLossV2(nn.Module):
    def __init__(
            self,
            adversarial_weight: float = 0.001,
            pixel_weight: float = 0.006,
            total_variation_weight: float = 2e-8,
            pixel_loss: str = "l2"
    ) -> None:
        super(GeneratorLossV2, self).__init__()
        vgg = models.vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False

        self.adversarial_weight: float = adversarial_weight
        self.pixel_weight: float = pixel_weight
        self.total_variation_weight: float = total_variation_weight

        self.feature_network = loss_network
        if pixel_loss == "l2":
            self.pixel_loss = nn.MSELoss()
        else:
            self.pixel_loss = nn.L1Loss()
        self.tv_loss = TVLoss()

    def forward(self, generated: torch.Tensor, target: torch.Tensor, fake_predictions: torch.Tensor) -> torch.Tensor:
        adversarial_loss = -torch.mean(torch.log(fake_predictions + 1e-8))
        perception_loss = self.pixel_loss(self.feature_network(generated), self.feature_network(target))
        image_loss = self.pixel_loss(generated, target)
        tv_loss = self.tv_loss(generated)

        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
