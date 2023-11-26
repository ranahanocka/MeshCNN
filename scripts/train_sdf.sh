#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/sdf \
--name sdf \
--ncf 64 128 256 256 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--dataset_mode regression \
--output_dimension 1 \
--num_samples 27000 \
--point_encode no_encode \
--lr 0.0001 \

