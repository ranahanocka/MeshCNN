#!/usr/bin/env bash

## run the training
python tuning.py \
--dataroot $(pwd)/datasets/roof_seg \
--name roof_seg \
--arch meshunet \
--dataset_mode segmentation \
--ninput_edges 16000 \
--pool_res 12000 10500 9000 \
--batch_size 1 \
--num_aug 20 \
--max_epochs 100
#--from_pretrained checkpoints/coseg_aliens/latest_net.pth