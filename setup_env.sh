GITHUB_ROOT="$(pwd)/pytorch-dense-correspondence-private"

# from entrypoint.sh
export DATA_DIR=$GITHUB_ROOT/data
export DC_DATA_DIR=$DATA_DIR/pdc
export DC_SOURCE_DIR=$GITHUB_ROOT
export PDC_BUILD_DIR=$DC_SOURCE_DIR/build
export POSER_BUILD_DIR=$PDC_BUILD_DIR/poser
export COCO_CUSTOM_DATA_DIR=$DC_DATA_DIR/coco

# from setup_environment.sh
export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR/modules
export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR
export PYTHONPATH=$PYTHONPATH:$DC_SOURCE_DIR/external/pytorch-segmentation-detection

export PATH=$PATH:$DC_SOURCE_DIR/bin
export PATH=$PATH:$DC_SOURCE_DIR/modules/dense_correspondence_manipulation/scripts
