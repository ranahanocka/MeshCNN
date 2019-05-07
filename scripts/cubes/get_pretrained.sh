#!/usr/bin/env bash

CHECKPOINT='checkpoints/cubes'

# get pretrained model
mkdir -p $CHECKPOINT
wget https://www.dropbox.com/s/fg7wum39bmlxr7w/cubes_wts.tar.gz
tar -xzvf cubes_wts.tar.gz && rm cubes_wts.tar.gz
mv latest_net.pth $CHECKPOINT
echo "downloaded pretrained weights to" $CHECKPOINT