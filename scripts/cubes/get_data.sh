#!/usr/bin/env bash

DATADIR='datasets' #location where data gets downloaded to

# get data
mkdir -p $DATADIR && cd $DATADIR
wget https://www.dropbox.com/s/2bxs5f9g60wa0wr/cubes.tar.gz
tar -xzvf cubes.tar.gz && rm cubes.tar.gz
echo "downloaded the data and put it in: " $DATADIR