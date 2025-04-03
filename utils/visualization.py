import matplotlib.pyplot as plt
import torch
import cv2


def visualize_images(input_img: torch.Tensor, generated_img: torch.Tensor, target_img: torch.Tensor) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Входное изображение', 'Сгенерированное изображение', 'Эталонное изображение']

    for ax, img, title in zip(axes, [input_img, generated_img, target_img], titles):
        ax.imshow(img[0].permute(1, 2, 0).cpu().numpy().astype('uint8'))
        ax.set_title(title)
        ax.axis('off')

    plt.show(block=False)
    plt.pause(1)


class MatplotlibDrawer(object):
    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.titles = ['Входное изображение', 'Сгенерированное изображение', 'Эталонное изображение']
        for ax, title in zip(self.axes, self.titles):
            ax.set_title(title)
            ax.axis('off')

    def visualize_images(self, input_img, generated_img, target_img):
        for ax, img in zip(self.axes, [input_img, generated_img, target_img]):
            # ax.imshow(cv2.cvtColor(img[0].permute(1, 2, 0).cpu().numpy().astype('uint8'), cv2.COLOR_BGR2RGB))
            ax.imshow(img[0].permute(1, 2, 0).cpu().numpy().astype('uint8'))

        plt.draw()
        plt.pause(1e-3)


class OpenCVDrawer(object):
    def __init__(self):
        cv2.namedWindow('Merge visualization', cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_KEEPRATIO)

    @staticmethod
    def visualize_images(input_img, generated_img, target_img):
        cv2.imshow('Merge visualization',
                   cv2.hconcat([
                       cv2.cvtColor(input_img[0].permute(1, 2, 0).cpu().numpy().astype('uint8'), cv2.COLOR_RGB2BGR),
                       cv2.cvtColor(generated_img[0].permute(1, 2, 0).cpu().numpy().astype('uint8'), cv2.COLOR_RGB2BGR),
                       cv2.cvtColor(target_img[0].permute(1, 2, 0).cpu().numpy().astype('uint8'), cv2.COLOR_RGB2BGR)
                   ]))
        cv2.waitKey(1)
