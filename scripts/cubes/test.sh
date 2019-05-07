#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/cubes \
--name cubes \
--ncf 64 128 256 256 \
--pool_res 600 450 300 210 \
--norm group \
--resblocks 1 \
--export_folder meshes \