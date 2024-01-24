## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common
from model import ops
import torch.nn as nn
import torch


def make_model(args, parent=False):
    return CARN(args)


class ConvBN(nn.Module):
    def __init__(
            self, n_feat):

        super(ConvBN, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, 3, padding=1))
        modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(nn.ReLU(True))
        modules_body.append(nn.Conv2d(n_feat, n_feat, 3, padding=1))
        modules_body.append(nn.BatchNorm2d(n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.ResidualBlock(64, 64)
        self.b2 = ops.ResidualBlock(64, 64)
        self.b3 = ops.ResidualBlock(64, 64)
        self.c1 = ops.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class CARN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(CARN, self).__init__()

        scale = args.scale[0]

        # RGB mean for DIV2K
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        #Single Image Mean
        # self.sub_mean = common.SubMeanStd
        # self.add_mean = common.AddMeanStd

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = Block(64, 64)
        self.b2 = Block(64, 64)
        self.b3 = Block(64, 64)
        self.c1 = ops.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64 * 4, 64, 1, 1, 0)

        self.denoiser = nn.Sequential(*[ConvBN(64) for _ in range(5)])

        self.upsample = common.Upsampler(conv, scale, 64, act=False)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        # rgb_mean, rgb_std = common.get_mean_std(x)
        x = self.sub_mean(x)
        # x = self.sub_mean(rgb_mean, rgb_std)(x)

        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3)

        out = self.denoiser(out)

        out = self.exit(out)
        out = self.add_mean(out)
        # out = self.add_mean(rgb_mean, rgb_std)(out)

        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
