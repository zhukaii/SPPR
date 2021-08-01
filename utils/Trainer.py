import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
import logging

from utils.model import prepare_model
from utils.data import datatypes
from utils.model_para import filter_para
from utils.loss import MyLosses
from utils.evaluate import Evaluator
from utils.lr import LR_Scheduler
from utils.lr import CategoriesSampler
from torch.utils.data import DataLoader
from collections import OrderedDict


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        # self.saver = Saver(args)
        # self.saver.save_experiment_config()

        # Define dataloader for train/test
        prepare_data = datatypes[args.dataset_type]
        self.dataset = prepare_data(args.dataset_name, args)
        self.label_per_task = [list(np.array(range(args.base_class)))] + [list(np.array(range(args.way)) +
                                                                               args.way * task_id + args.base_class)
                                                                          for task_id in range(args.tasks)]

        # Define model and optimizer
        model = prepare_model(args)
        optim_para = filter_para(model, args)
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(optim_para, weight_decay=args.weight_decay, nesterov=args.nesterov)
        else:
            optimizer = torch.optim.Adam(optim_para)

        # Define Criterion
        # whether to use class balanced weights
        # if args.use_balanced_weights:
        #     weight = balance_weights()
        # else:
        #     weight = None
        self.criterion = MyLosses(weight=None, args=self.args).build_loss(mode=args.loss_type)

        # Define Evaluator
        self.evaluator = Evaluator(args.all_class)

        # Using cuda
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        if torch.cuda.is_available():
            model = model.cuda()
        self.model, self.optimizer = model, optimizer
        self.model.train()

        # Resuming checkpoint
        self.best_pred = 0.0
        # if args.resume is not None:
        #     resume()
        #
        # # Clear start epoch if fine-tuning
        # if args.ft:
        #     args.start_epoch = 0

        # history of prediction
        self.acc_history = []

    def training(self, session):
        # self.model.train()
        train_dataset_new = self.dataset[0][session]
        session_class = self.args.base_class + self.args.way * session
        session_class_last = self.args.base_class + self.args.way * (session - 1)
        epochs = self.args.base_epochs
        train_dataset = train_dataset_new
        if session == 0:
            if self.args.val:
                para = torch.load(self.args.model_path)
                self.model.load_state_dict(para)
            else:
                train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4,
                                          pin_memory=True)
                train_sampler = CategoriesSampler(train_dataset.sub_indexes,
                                                  len(train_loader),
                                                  self.args.way+3,
                                                  self.args.shot)
                train_fsl_loader = DataLoader(dataset=train_dataset,
                                              batch_sampler=train_sampler,
                                              num_workers=4,
                                              pin_memory=True)
                self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr, epochs, len(train_loader), args=self.args)
                for epoch in range(epochs):
                    tbar = tqdm(train_loader)
                    train_loss = 0.0
                    for i, sample in enumerate(zip(tbar, train_fsl_loader)):
                        query_image, query_target = sample[0][0], sample[0][1]
                        support_image, support_target = sample[1][0], sample[1][1]
                        if self.args.cuda:
                            query_image, query_target, support_image, support_target = query_image.cuda(), query_target.cuda(), support_image.cuda(), support_target.cuda()
                        if not self.args.no_lr_scheduler:
                            self.scheduler(self.optimizer, i, epoch, self.best_pred)
                        self.optimizer.zero_grad()
                        output = self.model(query_image, support_image, support_target)[:, :session_class]

                        loss = self.criterion(output, query_target.view(-1, 1).repeat(1, 3).view(-1), session_class_last)
                        loss.backward()
                        self.optimizer.step()
                        train_loss += loss.item()

                    print('[Session: %d, Epoch: %d, numImages: %5d]' % (
                        session, epoch, i * self.args.batch_size + query_image.data.shape[0]))
                    print('Loss: %.3f' % (train_loss / (i + 1)))

                torch.save(self.model.state_dict(),
                           os.path.join(self.args.dir_name, 'model_session_0_'+self.args.dataset_name+'.pth'))

        else:
            train_loader = DataLoader(train_dataset, batch_size=250, shuffle=False, num_workers=4,
                                      pin_memory=True)
            if self.args.dataset_name == 'cub':
                embeddings = []
                for i, sample in enumerate(train_loader):
                    image, target = sample[0], sample[1]
                    if self.args.cuda:
                        image, target = image.cuda(), target.cuda()
                    with torch.no_grad():
                        a = self.model.calculate_means_cub1(image)
                        embeddings.append(a)
                embeddings = torch.cat(embeddings, dim=0)
                with torch.no_grad():
                    self.model.calculate_means_cub2(embeddings)
            else:
                for i, sample in enumerate(train_loader):
                    image, target = sample[0], sample[1]
                    if self.args.cuda:
                        image, target = image.cuda(), target.cuda()
                    with torch.no_grad():
                        self.model.calculate_means(image)

    def validation(self, session):
        self.model.eval()
        self.evaluator.reset()
        test_dataset = self.dataset[1][session]
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        session_class = self.args.base_class + self.args.way * session
        for i, sample in enumerate(test_loader):
            image, target = sample[0], sample[1]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model.forward_test(image)[:, :session_class]
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Test
        _, Acc_class, Acc = self.evaluator.Acc(session_class)
        self.acc_history.append(Acc)
        logging.info('Validation:')
        logging.info('[Session: %d, numImages: %5d]' % (session, i * self.args.batch_size + image.data.shape[0]))
        logging.info("Acc_class:{}, Acc:{} \n".format(Acc_class, Acc))
