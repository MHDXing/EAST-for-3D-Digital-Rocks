import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numbers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Charbonnier_loss(torch.nn.Module):
    """Charbonnier Loss"""
    def __init__(self):
        super(Charbonnier_loss, self).__init__()
        self.eps = 1e-6    # epsilon^2
 
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


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
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding, self.padding, self.padding), mode='reflect')  # reflect,replicate
        return input - self.conv(x, weight=self.weight, groups=self.groups)


class HighFrequency_loss(torch.nn.Module):
    """High Frequency Loss"""
    def __init__(self):
        super(HighFrequency_loss, self).__init__()
        
        self.highfreqcy_1 = GaussianSmoothing(1, 5, 5, dim=3)
        self.highfreqcy_2 = GaussianSmoothing(1, 5, 5, dim=3)

    def forward(self, X, Y):
        X = self.highfreqcy_1(X)
        Y = self.highfreqcy_2(Y)
        diff = torch.add(X, -Y)
        loss = torch.mean(torch.abs(diff))
        return loss


class GW_loss(torch.nn.Module):
    '''Gradient Weighted Loss'''
    def __init__(self):
        super(GW_loss, self).__init__()

    def forward(self, x1, x2):
        # sobel_x = torch.Tensor([[-1,0,1],[-2,0,2],[-1,0,1]])
        sobel_x = torch.Tensor([[[-1,0,1],[-1,0,1],[-1,0,1]], 
                                [[-1,0,1],[-2,0,2],[-1,0,1]],
                                [[-1,0,1],[-1,0,1],[-1,0,1]]])
        b, c, d, h, w = x1.shape
        sobel_x = sobel_x.expand(c, 1, 3, 3, 3) / 255
        sobel_y = sobel_x.permute(0, 1, 2, 4, 3)
        sobel_z = sobel_x.permute(0, 1, 4, 3, 2)
        sobel_x = sobel_x.type_as(x1)
        sobel_y = sobel_y.type_as(x1)
        sobel_z = sobel_z.type_as(x1)
        weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
        weight_y = nn.Parameter(data=sobel_y, requires_grad=False)
        weight_z = nn.Parameter(data=sobel_z, requires_grad=False)

        Ix1 = F.conv3d(x1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv3d(x2, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv3d(x1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv3d(x2, weight_y, stride=1, padding=1, groups=c)
        Iz1 = F.conv3d(x1, weight_z, stride=1, padding=1, groups=c)
        Iz2 = F.conv3d(x2, weight_z, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        dz = torch.abs(Iz1 - Iz2)
        # loss = (1 + 2*dx) * (1 + 2*dy) * (1 + 2*dz) * torch.abs(x1 - x2)
        loss = (1 + 4 * dx*dy*dz) * torch.abs(x1 - x2)
        return torch.mean(loss)


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'Charbonnier':
                loss_function = Charbonnier_loss()
            elif loss_type == 'HF':
                loss_function = HighFrequency_loss()
            elif loss_type == 'GW':
                loss_function = GW_loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        # 20231102
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
        return loss_sum
        # return losses

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

