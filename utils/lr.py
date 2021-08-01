import math
import random
import torch


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=30, warmup_epochs=0, args=None):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        self.args = args
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.5 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        # if epoch > self.epoch:
        #     print('\n=>Epoches %i, learning rate = %.4f, \
        #         previous best = %.4f' % (epoch, lr, best_pred))
        #     self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        para_len = len(optimizer.param_groups)
        if para_len == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            for i in range(para_len):
                optimizer.param_groups[i]['lr'] = lr * self.args.lr_coefficient[i]
            # for i in range(1, len(optimizer.param_groups)):
            #     optimizer.param_groups[i]['lr'] = lr * 10


class CategoriesSampler:

    def __init__(self, index, lenth, way, shot):
        self.lenth = lenth
        self.way = way
        self.shot = shot

        self.index = index

    def __len__(self):
        return self.lenth

    def __iter__(self):
        for lenth in range(self.lenth):
            batch = []
            num_class = list(self.index.keys())
            random.shuffle(num_class)
            classes = num_class[:self.way]
            for c in classes:
                lenth_per = torch.from_numpy(self.index[c])
                way_per = len(lenth_per)
                shot = torch.randperm(way_per)[:self.shot]
                batch.append((c*way_per+shot).int())
            batch = torch.stack(batch).reshape(-1)
            yield batch