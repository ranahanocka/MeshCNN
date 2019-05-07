#!/usr/bin/env bash

CHECKPOINT='checkpoints/human_seg'
mkdir -p $CHECKPOINT

wget https://www.dropbox.com/s/8i26y7cpi6st2ra/human_seg_wts.tar.gz
tar -xzvf human_seg_wts.tar.gz && rm human_seg_wts.tar.gz
mv latest_net.pth $CHECKPOINT
echo "downloaded pretrained weights to" $CHECKPOINT
