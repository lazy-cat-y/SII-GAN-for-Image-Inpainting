import torch

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torch.nn.functional import conv2d


class VGG19(nn.Module):
    def __init__(self, resize_input=False):
        super(VGG19, self).__init__()

        features = models.vgg19(pretrained=True).features
        self.resize_input = resize_input
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

        prefix = [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        posfix = [1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        names = list(zip(prefix, posfix))

        self.relus = []
        for pre, pos in names:
            self.relus.append('relu{}_{}'.format(pre, pos))
            self.__setattr__('relu{}_{}'.format(pre, pos), nn.Sequential())

        nums = [[0, 1], [2, 3], [4, 5, 6], [7, 8],
                [9, 10, 11], [12, 13], [14, 15], [16, 17],
                [18, 19, 20], [21, 22], [23, 24], [25, 26],
                [27, 28, 29], [30, 31], [32, 33], [34, 35]]

        for i, layer in enumerate(self.relus):
            for num in nums[i]:
                self.__getattr__(layer).add_module(str(num), features[num])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # resize and normalize input for pretrained vgg19
        x = (x + 1.0) / 2.0
        x = (x - self.mean.view(1, 3, 1, 1)) / (self.std.view(1, 3, 1, 1))
        if self.resize_input:
            x = F.interpolate(
                x, size=(256, 256), mode='bilinear', align_corners=True)
        features = []
        for layer in self.relus:
            x = self.__getattr__(layer)(x)
            features.append(x)
        out = {key: value for (key, value) in list(zip(self.relus, features))}
        return out


def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack([torch.exp(torch.tensor(gauss_fcn(x)))
                         for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
        raise TypeError(
            "kernel_size must be an odd positive integer. Got {}".format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(kernel_size, sigma):
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma))
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._padding = self.compute_zero_padding(kernel_size)
        self.kernel = get_gaussian_kernel2d(kernel_size, sigma)

    @staticmethod
    def compute_zero_padding(kernel_size):
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def forward(self, x):
        if not torch.is_tensor(x):
            raise TypeError(
                "Input x type is not a torch.Tensor. Got {}".format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError(
                "Invalid input shape, we expect BxCxHxW. Got: {}".format(x.shape))

        b, c, h, w = x.shape
        tmp_kernel: torch.Tensor = self.kernel.to(x.device).to(x.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        return conv2d(x, kernel, padding=self._padding, stride=1, groups=c)


def gaussian_blur(input, kernel_size, sigma):
    return GaussianBlur(kernel_size, sigma)(input)
