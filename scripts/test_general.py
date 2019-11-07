import pytest
import os
import shutil
import glob
import subprocess
'''
scripts for unit testing
'''


def get_data(dset):
    dpaths = glob.glob('./datasets/{}*'.format(dset))
    [shutil.rmtree(d) for d in dpaths]
    cmd = './scripts/{}/get_data.sh > /dev/null 2>&1'.format(dset)
    os.system(cmd)

def temp_file_name(dset):
    return './scripts/{}/train_temp.sh'.format(dset)

def run_train(dset):
    train_file = './scripts/{}/train.sh'.format(dset)
    temp_train_file = temp_file_name(dset)
    p = subprocess.run(['cp', '-p', '--preserve', train_file, temp_train_file])
    with open(train_file) as f:
        tokens = f.readlines()
    # now make the config so it only trains for one iteration
    tokens[-1] = tokens[-1] + '\n'
    tokens.append('--niter_decay 0 \\\n')
    tokens.append('--niter 1 \\')
    with open(temp_train_file, 'w') as f:
        f.writelines(tokens)
    cmd = "bash -c 'source ~/anaconda3/bin/activate ~/anaconda3/envs/meshcnn && {}'".format(temp_train_file)
    os.system(cmd)
    os.remove(temp_train_file)

def get_pretrained(dset):
    cmd = './scripts/{}/get_pretrained.sh > /dev/null 2>&1'.format(dset)
    os.system(cmd)

def run_test(dset):
    cmd = "bash -c 'source ~/anaconda3/bin/activate ~/anaconda3/envs/meshcnn && ./scripts/{}/test.sh'".format(dset)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (_out, err) = proc.communicate()
    out = str(_out)
    idf0 = 'TEST ACC: ['
    token = out[out.find(idf0) + len(idf0):]
    idf1 = '%]'
    accs = token[:token.find(idf1)]
    acc = float(accs)
    if dset == 'shrec':
        assert acc == 99.167, "shrec accuracy was {} and not 99.167".format(acc)

def test_one():
    dsets = ['shrec']
    for dset in dsets:
        get_data(dset)
        run_train(dset)
        get_pretrained(dset)
        run_test(dset)