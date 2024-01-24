import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.input_large = (args.model == 'VDSR')
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()

        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def forward_chop(self, x, shave=4, min_size=360000):  # shave=10,patch64=4  16w
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, d, h, w = x.size()
        d_half, h_half, w_half = d // 2, h // 2, w // 2  # 25//2=12
        d_size, h_size, w_size = d_half + shave, h_half + shave, w_half + shave  # 多扩充一部分避免边界影响,16

        lr_list = [
            x[:, :, 0:d_size, 0:h_size, 0:w_size],                     # part 1, 向后裁切
            x[:, :, 0:d_size, 0:h_size, (w - w_size):w],               # part 2, 向前裁切
            x[:, :, 0:d_size, (h - h_size):h, 0:w_size],               # patr 3
            x[:, :, 0:d_size, (h - h_size):h, (w - w_size):w],
            x[:, :, (d - d_size):d, 0:h_size, 0:w_size],
            x[:, :, (d - d_size):d, 0:h_size, (w - w_size):w],
            x[:, :, (d - d_size):d, (h - h_size):h, 0:w_size],
            x[:, :, (d - d_size):d, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size * d_size < min_size:      # if sub_cub < min_size
            sr_list = []
            for i in range(0, 8, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)  # 
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        d, h, w = scale * d, scale * h, scale * w
        d_half, h_half, w_half = scale * d_half, scale * h_half, scale * w_half
        d_size, h_size, w_size = scale * d_size, scale * h_size, scale * w_size
        # shave *= scale

        output = x.new(b, c, d, h, w)
        output[:, :, 0:d_half, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:d_half, 0:h_half, 0:w_half]
        output[:, :, 0:d_half, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:d_half, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, 0:d_half, h_half:h, 0:w_half] \
            = sr_list[2][:, :, 0:d_half, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, 0:d_half, h_half:h, w_half:w] \
            = sr_list[3][:, :, 0:d_half, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        output[:, :, d_half:d, 0:h_half, 0:w_half] \
            = sr_list[4][:, :, (d_size - d + d_half):d_size, 0:h_half, 0:w_half]
        output[:, :, d_half:d, 0:h_half, w_half:w] \
            = sr_list[5][:, :, (d_size - d + d_half):d_size, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, d_half:d, h_half:h, 0:w_half] \
            = sr_list[6][:, :, (d_size - d + d_half):d_size, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, d_half:d, h_half:h, w_half:w] \
            = sr_list[7][:, :, (d_size - d + d_half):d_size, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y
