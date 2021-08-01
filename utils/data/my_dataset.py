from torch.utils.data import Dataset
from collections import defaultdict
from utils.data.para import AVAILABLE_TRANSFORMS_train, AVAILABLE_TRANSFORMS_test

import numpy as np
import random


def ILdataset_shot(dataset, args):
    label_per_task = [list(np.array(range(args.base_class)))] + [list(np.array(range(args.way)) +
                                                                      args.way * task_id + args.base_class) for task_id
                                                                 in range(args.tasks)]
    dataset_train = dataset('{dir}/{name}'.format(dir=args.data_dir, name=args.dataset_name),
                                     train=True, transform=AVAILABLE_TRANSFORMS_train[args.dataset_name],
                                     target_transform=None, args=args, download=True)
    dataset_test = dataset('{dir}/{name}'.format(dir=args.data_dir, name=args.dataset_name),
                                    train=False, transform=AVAILABLE_TRANSFORMS_test[args.dataset_name],
                                    target_transform=None, args=args, download=True)

    train_datasets = []
    test_datasets = []

    for task_id in range(args.session):
        train_datasets.append(SubData_train(dataset_train, label_per_task, args, task_id))
        test_datasets.append(SubData_test(dataset_test, label_per_task, task_id))

    return [train_datasets, test_datasets]


class SingleClass(Dataset):
    def __init__(self, dataset, sublabel):
        self.ds = dataset.ds
        self.sub_indexes = dataset.sub_indexes[int(sublabel)]

    def __getitem__(self, item):
        return self.ds[self.sub_indexes[item]]

    def __len__(self):
        return len(self.sub_indexes)


class SubData_train(Dataset):
    def __init__(self, dataset, sublabels, args, task_ids):
        self.ds = dataset
        self.indexes = []
        self.sub_indexes = defaultdict(list)
        if task_ids == 0:
            sublabel = sublabels[task_ids]
            for label in sublabel:
                self.indexes.extend(dataset.sub_indexes[int(label)])
                self.sub_indexes[label] = dataset.sub_indexes[int(label)]
        else:
            for task in range(1, task_ids+1):
                sublabel = sublabels[task]
                for label in sublabel:
                    # shot_sample = list(dataset.sub_indexes[int(label)])[0:5]
                    # shot_sample = dataset.sub_indexes[label]
                    shot_sample = random.sample(list(dataset.sub_indexes[int(label)]), args.shot)
                    self.indexes.extend(shot_sample)
                    self.sub_indexes[label] = shot_sample

    def __getitem__(self, item):
        return self.ds[self.indexes[item]]

    def __len__(self):
        return len(self.indexes)


class SubData_test(Dataset):
    def __init__(self, dataset, sublabels, task_ids):
        self.ds = dataset
        self.sub_indexes = []
        for task in range(task_ids+1):
            sublabel = sublabels[task]
            for label in sublabel:
                self.sub_indexes.extend(dataset.sub_indexes[int(label)])

    def __getitem__(self, item):
        return self.ds[self.sub_indexes[item]]

    def __len__(self):
        return len(self.sub_indexes)