#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/human_seg \
--name human_seg \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 3000 \
--pool_res 2000 1000 500 \
--num_threads 0 \
--resblocks 1 \
--batch_size 2 \
--lr 0.001 \
--num_aug 20 \
--slide_verts 0.2 \
--gpu_ids -1 \
--verbose_plot \
--weighted_loss 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125 \
