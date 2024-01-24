import random
import numbers
import numpy as np
import skimage.color as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        
        self.padding = kernel_size // 2

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        # print(kernel)

        self.register_buffer('weight', kernel)
        self.groups = channels
        

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding, self.padding, self.padding), mode='replicate')  # reflect,replicate
        return self.conv(x, weight=self.weight, groups=self.groups)

def get_patch(*args, patch_size=96, scale=2, multi=False):

    ic, ih, iw = args[0].shape

    p = scale if multi else 1
    tp = p * patch_size
    ip = tp // scale

    iz = random.randint(0, ic - ip)
    ix = random.randint(0, ih - ip)
    iy = random.randint(0, iw - ip)

    tz, tx, ty = scale * iz, scale * ix, scale * iy

    ret = [
        args[0][iz:iz + ip, ix:ix + ip, iy:iy + ip],
        *[a[tz:tz + tp, tx:tx + tp, ty:ty + tp] for a in args[1:]]
    ]
    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):

        tensor = torch.from_numpy(np.ascontiguousarray(img)).float()
        tensor.mul_(rgb_range / 255)

        return torch.unsqueeze(tensor, 0)

    return [_np2Tensor(a) for a in args]

def augment(*args, flip=True, rot=True):
    if flip:
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        cflip = random.random() < 0.5
    # hflip = flip and random.random() < 0.5
    # rot90 = rot and random.random() < 0.5

    # if rot: rot = random.randint(0, 5) # for EAST
    if rot: rot = random.randint(0, 1)
    # 存储转置矩阵
    rot_matrix = {
        1: (0, 2, 1),
        2: (1, 0, 2),
        3: (1, 2, 0),
        4: (2, 0, 1),
        5: (2, 1, 0)
    }

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[:, :, ::-1]
        if cflip: img = img[::-1, :, :]  # for EAST

        if rot: img = img.transpose(rot_matrix[rot]) # for EAST
        # if rot: img = img.transpose(rot_matrix[2])
        return img

    return [_augment(a) for a in args]


def add_noise(img, zoom=[0.5, 0.05]):

    # random guassian blur
    img = torch.tensor(np.ascontiguousarray(img))  # numpy to tensor
    blur = torch.unsqueeze(torch.unsqueeze(img, 0), 0).float()  # 增加两个维度，double转float，pytorch默认32位浮点
    sigma = random.random()*zoom[0]
    blur = GaussianSmoothing(1, 3, sigma, dim=3)(blur)
    img = blur.numpy()[0][0]

    # Poisson Noise
    # vals = len(np.unique(img)) * zoom  # grayscale value distribution range
    # vals = 2 ** np.ceil(np.log2(vals))
    vals = np.median(img)*zoom[1]
    # add noise, minus mean to keep the image intensity
    noise = np.random.poisson(vals, img.shape) - vals
    img = np.clip(img+noise, a_min=0.0, a_max=255.0)

    return img