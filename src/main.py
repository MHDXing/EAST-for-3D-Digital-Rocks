import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def main():
    global model
    start_time = datetime.datetime.now()
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            epoch = 0  # modify
            while not t.terminate():
                t.train()
                epoch += 1
                if epoch % args.test_every == 0:
                # if epoch % args.test_every == 0 and epoch>=505:  # continue
                    t.test()

            checkpoint.done()
    end_time = datetime.datetime.now()
    print('Running time is:', end_time-start_time)

if __name__ == '__main__':
    main()
