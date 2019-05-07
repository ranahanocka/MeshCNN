#!/usr/bin/env bash

DATADIR='datasets' #location where data gets downloaded to

# get data
mkdir -p $DATADIR && cd $DATADIR
wget https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz
tar -xzvf shrec_16.tar.gz && rm shrec_16.tar.gz
echo "downloaded the data and putting it in: " $DATADIR
