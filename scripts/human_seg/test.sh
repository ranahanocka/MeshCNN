#!/usr/bin/env bash

## run the test and export collapses
python3 test.py \
--dataroot datasets/human_seg \
--name human_seg \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 3000 \
--pool_res 2000 1000 500 \
--num_threads 0 \
--resblocks 1 \
--batch_size 12 \
--export_folder meshes \
--gpu_ids -1 \
