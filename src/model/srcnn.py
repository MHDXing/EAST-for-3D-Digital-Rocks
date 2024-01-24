from model import common

import torch.nn as nn

def make_model(args, parent=False):
    return SRCNN(args)


## PreUpsample
class PreUpsample(nn.Module):
    def __init__(self, scale=2):
        super(PreUpsample, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=(scale, scale, scale), mode='trilinear', align_corners=False)  # align_corners=False

    def forward(self, x):
        x = self.upsample(x)
        return x


## SRCNN
class SRCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRCNN, self).__init__()
        
        n_resblocks = args.n_resblocks
        # n_resblocks = 12
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean
        self.sub_mean = common.MeanShift(args.rgb_range)

        self.Upsampler = PreUpsample(scale)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        # define body module
        modules_body = [conv(n_feats, n_feats, kernel_size) for _ in range(n_resblocks-2)]
        modules_body.append(conv(n_feats, args.n_colors, kernel_size))

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        
        x = self.Upsampler(x)
        x = self.sub_mean(x)

        res = self.head(x)
        res = self.body(res)
        res += x

        x = self.add_mean(res)

        return x 

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
