# occnet

Automatic Keypoint Discovery with Occlusions

# Structure

- [data/](data)

    This folder contains code for using the `pytorch-dense-correspondence-private` repo for data loading. This code is used to format data into the correct form for occnet. pytorch-dense-correspondence-private is used because it has real-world data that matches the format we need for occnet.

- [datasets](datasets)

    This folder is used to hold the datasets that we make. Folders within the datasets folder will be numbered `000`, `001`, `002`, etc.

- [experiments](experiments)

    This folder will contain the experiments and corresponding TensorBoard sessions. The folders will be named according to the time of experiment, with a timestamp of the form `YY-MM-DD_HH:MM:SS`.

- [evaluations](evaluations)

    This folder will hold the results of running an experiment on a selected dataset. The experiment and dataset result will be in a folder name with name `checkpoint_name###experiment###dataset`. For example, `model.ckpt-9576###19-03-23_18:27:59###000`.

- [notebooks](notebooks)

    This folder contains the .ipynb notebooks for experimentation, santity checks, or anything that is more convenient in a notebook than a standalone script.

- [maskrcnn](maskrcnn)

    This will hold the code needed to synthetically create COCO formatted datasets for training with Mask R-CNN. This allows us to create the binary instance masks at runtime.

# Environment Setup

```
# setup the environment
cd occnet
source setup_env.sh

# set up submodules
cd pytorch-dense-correspondence
git submodule update --init --recursive

# use dense object nets to download data
cd pytorch-dense-correspondence-private
mkdir data
cd data/
python ../config/download_pdc_data.py config/dense_correspondence/dataset/composite/caterpillar_only.yaml
# move the pdc folder to pytorch-dense-correspondence/data
mkdir data
mv pdc/ data/

# conda environment setup
conda create -n occnet python=3.7.2
conda activate occnet
# install dependencies
conda install pytorch torchvision -c pytorch
conda install ipykernel
conda install pyyaml
conda install matplotlib
conda install -c conda-forge opencv
conda install -c open3d-admin open3d
conda install -c anaconda tensorflow-gpu
conda install -c anaconda scipy
pip install pillow

# dependencies for the server
conda install -c anaconda flask
export FLASK_ENV=development
cd server/
export FLASK_APP=server.py
flask run --host=0.0.0.0

# for accessing google sheets
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
cd server/ (you will have to download a json file to use the Google Sheets API)
export GOOGLE_APPLICATION_CREDENTIALS=Occnet-869cc2aa84e8.json

# install kernel for jupyter notebook
python -m ipykernel install --user --name occnet --display-name "occnet"

# to start the notebook
cd /path/to/repo
jupyter notebook

# to access remotely
jupyter notebook --ip 0.0.0.0 --port 8888

# use the correct kernel (in web GUI)
Kernel -> Change kernel -> occnet

# make opencv in python work with window display support
pip install opencv-python 
pip install opencv-contrib-python
```