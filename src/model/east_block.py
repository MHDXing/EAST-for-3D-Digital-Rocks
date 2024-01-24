import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from model import common





class LFE(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='gelu'):  # act_type='gelu',gelu只改一处
        super(LFE, self).__init__()    
        self.exp_ratio = exp_ratio
        self.act_type  = act_type

        self.layernorm = nn.LayerNorm(inp_channels)
        # self.conv = nn.Conv3d(inp_channels, out_channels, kernel_size=1, padding=0)
        self.conv0 = nn.Conv3d(inp_channels, out_channels*exp_ratio, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv3d(out_channels*exp_ratio, out_channels, kernel_size=1, stride=1, padding=0)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        elif self.act_type == 'silu':
            self.act = common.SiLU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        x = self.layernorm(x.permute(0, 2, 3, 4, 1))  # batch,channel,H,W,D-->batch,H,W,D,channel
        x = x.permute(0, 4, 1, 2, 3).contiguous() # back
        # y = self.conv(x)
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)
        # y = 0.5 * y + 0.5 * y.mean(dim=1, keepdim=True) # 没用
        return y
    

class GMSA(nn.Module):
    def __init__(self, channels, shifts=0, window_sizes=[2, 4, 8]):
        super(GMSA, self).__init__()    
        self.channels = channels
        self.shifts = shifts
        self.window_sizes = window_sizes

        self.layernorm = nn.LayerNorm(self.channels)

        sum_channel = channels
        self.project_inp = nn.Conv3d(self.channels, sum_channel*3, kernel_size=1)  # *(q,k,v)
        self.split_chns  = [sum_channel, sum_channel, sum_channel]  # sum_channel*3/3 --> /(qkv*multihead)
        self.project_out = nn.Conv3d(sum_channel, self.channels, kernel_size=1)  # /qkv*3

    def forward(self, x):
        x = self.layernorm(x.permute(0, 2, 3, 4, 1))  # batch,channel,H,W,D-->batch,H,W,D,channel
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        b, c, d, h, w = x.shape
        multi_head = 2
        # sqrt_d = math.sqrt(c / multi_head / 3)
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []

        for idx, x_ in enumerate(xs):
            wsize = self.window_sizes[idx]

            if self.shifts == 1:
                x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2, -wsize//2), dims=(2,3,4))
            elif self.shifts == 2:
                x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
            elif self.shifts == 3:
                x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,4))
            elif self.shifts == 4:
                x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(3,4))

            q, k, v = rearrange(
                x_, 'b (qkv mulhead c) (d dd) (h dh) (w dw) -> qkv (b d h w) mulhead (dd dh dw) c', 
                qkv=3, mulhead=multi_head, dd=wsize, dh=wsize, dw=wsize
            )
            # atn = (q @ k.transpose(-2, -1)/sqrt_d)
            atn = (q @ k.transpose(-2, -1))
            atn = atn.softmax(dim=-1)
            y_ = (atn @ v)
            y_ = rearrange(
                y_, '(b d h w) mulhead (dd dh dw) c-> b (mulhead c) (d dd) (h dh) (w dw)', 
                mulhead=multi_head, d=d//wsize, h=h//wsize, w=w//wsize, dd=wsize, dh=wsize, dw=wsize
            )

            if self.shifts == 1:
                y_ = torch.roll(y_, shifts=(wsize//2, wsize//2, wsize//2), dims=(2,3,4))
            elif self.shifts == 2:
                y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2,3))
            elif self.shifts == 3:
                y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2,4))
            elif self.shifts == 4:
                y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(3,4))

            ys.append(y_)
            
        y = torch.cat(ys, dim=1)
        y = self.project_out(y)
        return y


class ELAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_sizes=[4, 8, 12]):
        super(ELAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        
        self.lfe = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        self.gmsa = GMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes)

    def forward(self, x):

        y= self.gmsa(x)
        y = y + x
        y = self.lfe(y) + y

        return y
