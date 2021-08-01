import torch.nn as nn
import torch


class MyLosses(object):
    def __init__(self, weight=None, reduction='mean', batch_average=False, ignore_index=255, args=None):
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.batch_average = batch_average
        self.cuda = args.cuda
        self.args = args

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        if mode == 'be':
            return self.BCELogitsLoss
        if mode == 'mse':
            return self.MSELoss

    def CrossEntropyLoss(self, logit, target, session_class_last):
        n, c = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def BCELogitsLoss(self, logit, target, session_class_last):
        weight = torch.ones_like(logit[0])
        if session_class_last >= self.args.base_class:
            weight[session_class_last:] = self.args.loss_weight
        # criterion = nn.BCEWithLogitsLoss(weight=self.weight, reduction=self.reduction)
        criterion = nn.BCELoss(weight=weight, reduction=self.reduction)

        onehot = torch.zeros(target.shape[0], session_class_last+self.args.way)
        if self.cuda:
            criterion = criterion.cuda()
            onehot = onehot.cuda()
        onehot.scatter_(dim=1, index=target.long().view(-1, 1), value=1.)

        loss = criterion(logit, onehot)

        return loss

    def MSELoss(self, logit, target, session_class_last):
        onehot = torch.zeros(target.shape[0], session_class_last + self.args.way)
        criterion = nn.MSELoss()
        if self.cuda:
            criterion = criterion.cuda()
            onehot = onehot.cuda()
        onehot.scatter_(dim=1, index=target.long().view(-1, 1), value=1.)

        loss = criterion(logit, onehot)

        return loss

