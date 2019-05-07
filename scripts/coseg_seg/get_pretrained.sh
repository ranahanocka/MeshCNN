#!/usr/bin/env bash

CHECKPOINT=checkpoints/coseg_aliens
mkdir -p $CHECKPOINT

#gets the pretrained weights
wget https://www.dropbox.com/s/er7my13k9dwg9ii/coseg_aliens_wts.tar.gz
tar -xzvf coseg_aliens_wts.tar.gz && rm coseg_aliens_wts.tar.gz
mv latest_net.pth $CHECKPOINT
echo "downloaded pretrained weights to" $CHECKPOINT