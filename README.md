# occnet

# Setup

```
cd occnet
source setup_env.sh
```

# Download Data

From the don tutorial, download the data.

```
cd pytorch-dense-correspondence-private
python config/download\_pdc\_data.py config/dense\_correspondence/dataset/composite/caterpillar\_only.yaml
```

move it do a data folder!

# Files

- data\_loader.py

This file is used to format the pytorch-dense-correspondence data into the correct format needed by occnet.


# Submodules

- models

I'm using a fork of tensorflow/models where I can modify the keypointnet work.

- pytorch-dense-correspondence-private

I'm also using a fork of pytorch-dense-correspondence. I'm using this repo for it's real-world data.


```
cd pytorch-dense-correspondence-private
git submodule update --init --recursive
```

environment setup (conda environment name = occnet)
```
conda install pytorch torchvision -c pytorch
conda install ipykernel
conda install pyyaml
conda install matplotlib
conda install -c conda-forge opencv
conda install -c open3d-admin open3d
conda install -c anaconda tensorflow-gpu
```
todo: add tensorflow

# use conda environment in jupyter notebook

```
# install kernel for jupyter notebook
python -m ipykernel install --user --name occnet --display-name "occnet"

# to start the notebook
cd /path/to/repo
jupyter notebook

# use the correct kernel (in web GUI)
Kernel -> Change kernel -> occnet
```


# Notes
- poser didn't clone correctly
- figure out how to get the submodules working on my own fork