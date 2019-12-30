#!/usr/bin/env bash

cd `dirname $0`

python train.py --dataset cifar10 --gpu_id 3 --experiment_dirpath ../experiments/dcgan/cifar10 --z_dim 100 --conditional --train_iters 50000 --show_every 2500 --k 1 --experiment_name "k=1"
python train.py --dataset cifar10 --gpu_id 3 --experiment_dirpath ../experiments/dcgan/cifar10 --z_dim 100 --conditional --train_iters 50000 --show_every 2500 --k 3 --experiment_name "k=3"
python train.py --dataset cifar10 --gpu_id 3 --experiment_dirpath ../experiments/dcgan/cifar10 --z_dim 100 --conditional --train_iters 50000 --show_every 2500 --k 3 --experiment_name "l2" --l2_loss

