from torch.utils.data import Dataset
import os
import warnings
from PIL import Image
import numpy as np
from collections import defaultdict


class MiniImageNet(Dataset):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None, args=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self._exts = ['.jpg', '.jpeg', '.png']
        if self.train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')

        self.data = []
        self.targets = []
        self.synsets = []
        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s' % (
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.data.append(filename)
                self.targets.append(label)

        if not args.no_order:
            self.targets = self._map_new_class_index(self.targets, args.orders)
        self.targets = np.array(self.targets)
        self.sub_indexes = defaultdict(list)
        target_max = np.max(self.targets)
        for i in range(target_max + 1):
            self.sub_indexes[i] = np.where(self.targets == i)[0]

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        target = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.targets)

    def _map_new_class_index(self, y, order):
        """Transforms targets for new class order."""
        return list(map(lambda x: order.index(x), y))

