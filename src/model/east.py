import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.east_block import ELAB
from model import common

def make_model(args, parent=False):
    return EAST(args)


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_du = nn.Sequential(
                nn.Conv3d(channels, channels // reduction, 1, padding=0, bias=True),
                # nn.ReLU(inplace=True),
                nn.GELU(),
                # common.SiLU(),
                nn.Conv3d(channels // reduction, channels, 1, padding=0, bias=True),
                nn.Sigmoid(),
        )

    def forward(self, x):
        avg = self.avg_pool(x)
        y = self.conv_du(avg)
        return x * y


class CABlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(CABlock, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),
                nn.GELU(),
                # common.SiLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.calayer = CALayer(out_channels, reduction=reduction)

    def forward(self, x):
        y = self.calayer(self.conv(x)) + x
        return y
    

class ResGroup(nn.Module):
    def __init__(self, n_resblocks, channels, r_expand, window_sizes, group_index):
        super(ResGroup, self).__init__()

        ca_channels = channels // 2
        res_group = []
        for j in range(n_resblocks):
            # z = 0 if j % 2 == 0 else group_index
            res_group.append(
                ELAB(channels, channels, r_expand, j%2*group_index, window_sizes)   # i % 2/5, noshift=0；奇数个不动，偶数个位移 j%2*group_index
                # ELAB(channels, channels, r_expand, 0, window_sizes)
            )
        res_group.append(nn.Conv3d(channels, ca_channels, kernel_size=1, padding=0))
        res_group.append(CABlock(ca_channels, ca_channels))
        res_group.append(nn.Conv3d(ca_channels, channels, kernel_size=1, padding=0))
        self.res_group = nn.Sequential(*res_group)

    def forward(self, x):
        res = self.res_group(x)
        return x + res


class EAST(nn.Module):
    def __init__(self, args):
        super(EAST, self).__init__()

        self.scale = args.scale[0]
        self.colors = args.n_colors

        self.window_sizes = list(map(lambda x: int(x), args.window_sizes.split('-')))     # args.window_sizes  [2,4,8],[4,8,8]
        self.m_east  = args.n_resblocks   # args.m_east   36
        self.g_east = args.n_resgroups    #
        self.c_east  = args.n_feats       # args.c_east  180
        self.r_expand = 1                 # args.r_expand
        

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [nn.Conv3d(self.colors, self.c_east, kernel_size=3, stride=1, padding=1)]

        # define body module
        m_body = []  # add cablock

        for i in range(self.g_east):
            m_body.append(ResGroup(self.m_east, self.c_east, self.r_expand, self.window_sizes, i%4+1))  # 1,2,3,4
        # m_body.append(nn.Conv3d(self.c_east, self.c_east, kernel_size=3, stride=1, padding=1))
        # m_body.append(nn.Conv3d(self.c_east, self.c_east, kernel_size=1, stride=1, padding=0))
        # m_body.append(CABlock(self.c_east, self.c_east, reduction=16))  # add cablock
        
        # define tail module
        m_tail = [nn.Conv3d(self.c_east, args.n_colors*self.scale**3, kernel_size=3, stride=1, padding=1),
                  common.PixelShuffle3d(self.scale)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        D, H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.tail(res)
        x = self.add_mean(x)

        return x[:, :, 0:D*self.scale, 0:H*self.scale, 0:W*self.scale]

    def check_image_size(self, x):
        _, _, d, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])

        mod_pad_d = (wsize - d % wsize) % wsize
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        # 'circular'循环填充123(123),'replicate'复制填充123(333),'reflect'反射填充123(321)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'replicate')
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
        
        
        # normmean = 49.6082 # 194.4797
        # # 修改偏置参数的值
        # submean = torch.full_like(own_state['sub_mean.bias'].data, -normmean)
        # own_state['sub_mean.bias'].copy_(submean)
        # # 修改偏置参数的值
        # addmean = torch.full_like(own_state['add_mean.bias'].data, normmean)
        # own_state['add_mean.bias'].copy_(addmean)
        # # print(own_state['sub_mean.bias'].requires_grad)
        # print(own_state['sub_mean.bias'])
