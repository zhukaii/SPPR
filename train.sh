#!/bin/sh

# cifar train
python train.py --backbone_name resnet18 --model_name my_new --dataset_type cl_shot --optim sgd --opt opt2 \
--loss_type ce --no_order --seed 1993 --way 5 --shot 5 --session 9 --lr 0.1 --lr-scheduler cos --batch_size 128 \
--gpu-ids 0  --base_epochs 56  --data_dir $your_path

## mini-imagenet train
#python train.py --backbone_name resnet18 --model_name my_new --dataset_name mini-imagenet --dataset_type cl_shot \
#--optim sgd --opt opt2 --loss_type ce --no_order --seed 1993 --way 5 --shot 5 --session 9 --lr 0.1 --lr-scheduler cos \
#--batch_size 128 --gpu-ids 0 --base_epochs 68 --data_dir $your_path
#sleep 5

## cub train
#python train.py --backbone_name resnet18 --model_name my_new --pretrained --dataset_name cub --dataset_type cl_shot \
#--optim sgd --opt opt2 --loss_type ce --no_order --seed 1993 --way 10 --shot 5 --session 11 --lr 0.1 --lr-scheduler cos \
#--batch_size 128 --gpu-ids 0 --base_class 100 --base_epochs 70 --data_dir $your_path
#sleep 5