#!/usr/bin/env bash

DATADIR='datasets' #location where data gets downloaded to

echo "downloading the data and putting it in: " $DATADIR
mkdir -p $DATADIR && cd $DATADIR
wget https://www.dropbox.com/s/34vy4o5fthhz77d/coseg.tar.gz
tar -xzvf coseg.tar.gz && rm coseg.tar.gz