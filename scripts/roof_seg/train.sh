#!/usr/bin/env bash

## run the training
python train_pl.py \
--dataroot datasets/roof_seg \
--name roof_seg \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 14000 \
--pool_res 12000 10500 9000 \
--resblocks 3 \
--batch_size 1 \
--lr 0.001 \
--num_aug 20 \
--slide_verts 0.2
#--from_pretrained checkpoints/human_seg/latest_net.pth