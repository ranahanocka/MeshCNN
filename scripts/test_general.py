import pytest
import os
import shutil
import glob
import subprocess

"""
scripts for unit testing
"""


def get_data(dset):
    dpaths = glob.glob("./datasets/{}*".format(dset))
    [shutil.rmtree(d) for d in dpaths]
    cmd = "./scripts/{}/get_data.sh > /dev/null 2>&1".format(dset)
    os.system(cmd)


def add_args(file, temp_file, new_args):
    with open(file) as f:
        tokens = f.readlines()
    # now make the config so it only trains for one iteration
    tokens[-1] = tokens[-1] + "\n"
    for arg in new_args:
        tokens.append(arg)
    with open(temp_file, "w") as f:
        f.writelines(tokens)


def run_train(dset):
    train_file = "./scripts/{}/train.sh".format(dset)
    temp_train_file = "./scripts/{}/train_temp.sh".format(dset)
    p = subprocess.run(["cp", "-p", "--preserve", train_file, temp_train_file])
    add_args(
        train_file,
        temp_train_file,
        [
            "--niter_decay 0 \\\n",
            "--niter 1 \\\n",
            "--max_dataset_size 2 \\\n",
            "--gpu_ids -1 \\",
        ],
    )
    cmd = "bash -c 'source ~/anaconda3/bin/activate ~/anaconda3/envs/meshcnn && {} > /dev/null 2>&1'".format(
        temp_train_file
    )
    os.system(cmd)
    os.remove(temp_train_file)


def get_pretrained(dset):
    cmd = "./scripts/{}/get_pretrained.sh > /dev/null 2>&1".format(dset)
    os.system(cmd)


def run_test(dset):
    test_file = "./scripts/{}/test.sh".format(dset)
    temp_test_file = "./scripts/{}/test_temp.sh".format(dset)
    p = subprocess.run(["cp", "-p", "--preserve", test_file, temp_test_file])
    add_args(test_file, temp_test_file, ["--gpu_ids -1 \\"])
    # now run inference
    cmd = "bash -c 'source ~/anaconda3/bin/activate ~/anaconda3/envs/meshcnn && {}'".format(
        temp_test_file
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (_out, err) = proc.communicate()
    out = str(_out)
    idf0 = "TEST ACC: ["
    token = out[out.find(idf0) + len(idf0) :]
    idf1 = "%]"
    accs = token[: token.find(idf1)]
    acc = float(accs)
    if dset == "shrec":
        assert acc == 99.167, "shrec accuracy was {} and not 99.167".format(acc)
    if dset == "human_seg":
        assert acc == 92.554, "human_seg accuracy was {} and not 92.554".format(acc)
    os.remove(temp_test_file)


def run_dataset(dset):
    get_data(dset)
    run_train(dset)
    get_pretrained(dset)
    run_test(dset)


def test_shrec():
    run_dataset("shrec")


def test_human_seg():
    run_dataset("human_seg")
