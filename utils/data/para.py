from torchvision import transforms
from utils.data.cifar import CIFAR100
from utils.data.mini_imagenet import MiniImageNet
from utils.data.cub import CUB


datasets_all = {
    'CIFAR100': CIFAR100,
    'mini-imagenet': MiniImageNet,
    'cub': CUB,
}

AVAILABLE_TRANSFORMS_train = {
    'CIFAR100': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
    ),
    'mini-imagenet': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
    ),
    'cub': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ),
}

AVAILABLE_TRANSFORMS_test = {
    'CIFAR100': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
    ),
    'mini-imagenet': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.472, 0.453, 0.410], [0.277, 0.268, 0.284])]
    ),
    'cub': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ),
}
