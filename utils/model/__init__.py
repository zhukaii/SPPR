from torchvision import models
from utils.model.my_model import my_model


my_models = {
    'my': my_model,
}

backbones = {
    'resnet18': models.resnet18,
}


def prepare_model(args):
    if args.model_name:
        model = my_models[args.model_name](backbone=backbones[args.backbone_name], pretrained=args.pretrained, args=args)
    else:
        model = None
    return model
