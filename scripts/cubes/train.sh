#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/cubes \
--name cubes \
--ncf 64 128 256 256 \
--pool_res 600 450 300 210 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \