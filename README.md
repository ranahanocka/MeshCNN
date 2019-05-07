<img src='docs/imgs/alien.gif' align="right" width=325>
<br><br><br>

# MeshCNN in PyTorch


### SIGGRAPH 2019 [[Paper]](https://bit.ly/meshcnn) <br>

MeshCNN is a general-purpose deep neural network for 3D triangular meshes, which can be used for tasks such as 3D shape classification or segmentation. This framework includes convolution, pooling and unpooling layers which are applied directly on the mesh edges.

<img src="docs/imgs/meshcnn_overview.png" align="center" width="750px"> <br>

The code was written by [Rana Hanocka](https://www.cs.tau.ac.il/~hanocka/) and [Amir Hertz](http://pxcm.org/) with support from [Noa Fish](http://www.cs.tau.ac.il/~noafish/).

# Getting Started

### Installation
- Clone this repo:
```bash
git clone https://github.com/ranahanocka/MeshCNN.git
cd MeshCNN
```
- Install dependencies: [PyTorch](https://pytorch.org/) version 1.0 (also works with 0.4+). <i> Optional </i>: [tensorboardX](https://github.com/lanpa/tensorboardX) for training plots.
  - Via new conda environment `conda env create -f environment.yml` (creates an environment called meshcnn)
  
### 3D Shape Classification on SHREC
Download the dataset
```bash
bash ./scripts/shrec/get_data.sh
```

Run training (if using conda env first activate env e.g. ```source activate meshcnn```)
```bash
bash ./scripts/shrec/train.sh
```

To view the training loss plots, in another terminal run ```tensorboard --logdir runs``` and click [http://localhost:6006](http://localhost:6006).

Run test and export the intermediate pooled meshes:
```bash
bash ./scripts/shrec/test.sh
```
An example of collapses for a mesh:

<img src="/docs/imgs/T252.png" width="450px"/> 

### 3D Shape Segmentation on Humans
Download the dataset
```bash
bash ./scripts/human_seg/get_data.sh
```

Run training
```bash
bash ./scripts/human_seg/train.sh
```
  
# Citation
If you find this code useful, please consider citing our paper
```
 @article{hanocka2019meshcnn,
  title={MeshCNN: A Network with an Edge},
  author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:1809.05910},
  year={2019}
}
```


# Contributing
Contributions to this repository are very welcome. Open an issue if you: have problems running the code, want to suggest improvements, or want to submit a pull request.
  
# Acknowledgments
This code design was adopted from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
