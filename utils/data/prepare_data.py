from utils.data.my_dataset import ILdataset_shot
from utils.data.para import datasets_all


def prepare_CL_shot(dataset_name, args):
    dataset = datasets_all[dataset_name]
    dataset = ILdataset_shot(dataset, args)
    return dataset
