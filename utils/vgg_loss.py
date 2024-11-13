import torch
import torchvision
from typing import List, Any
from torchvision.models import vgg19


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize: bool = True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = [torchvision.models.vgg16(pretrained=True).features[:4].cuda().eval(),
                  torchvision.models.vgg16(pretrained=True).features[4:9].cuda().eval(),
                  torchvision.models.vgg16(pretrained=True).features[9:16].cuda().eval(),
                  torchvision.models.vgg16(pretrained=True).features[16:23].cuda().eval()]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

    def forward(self, input: torch.Tensor, target: torch.Tensor, feature_layers: List[int] = [0, 1, 2, 3],
                style_layers: List[Any] = []) -> torch.Tensor:
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


def total_variation_loss(feature: torch.Tensor, weight: float = 1) -> float:
    bs_img, c_img, h_img, w_img = feature.size()
    tv_h = torch.pow(feature[:, :, 1:, :] - feature[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(feature[:, :, :, 1:] - feature[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


class VGGLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:25].eval().cuda()
        self.loss = torch.nn.MSELoss(size_average=11)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        vgg_first = self.vgg(input)
        vgg_second = self.vgg(target)
        perceptual_loss = self.loss(vgg_first, vgg_second)
        return perceptual_loss
