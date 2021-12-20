#!/usr/bin/env bash

## run the training
python tuning.py \
--dataroot $(pwd)/datasets/roof_seg \
--name roof_seg \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 16000 \
--pool_res 12000 10500 9000 \
--resblocks 2 \
--batch_size 1 \
--lr 0.01 \
--num_aug 20 \
--slide_verts 0.2 \
--warmup_epochs 70 \
--max_epochs 100
#--from_pretrained checkpoints/coseg_aliens/latest_net.pth