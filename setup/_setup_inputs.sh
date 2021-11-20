#!/usr/bin/env bash

set -e

conda install -yc anaconda wget
conda install -yc conda-forge unzip

HOME_DIR=$1/braincode/
cd $HOME_DIR

wget -O inputs.zip https://www.dropbox.com/s/63b16bx3hrx0rar/inputs.zip?dl=0
unzip inputs.zip -d inputs

# Process and populate benchmark metrics on input files
python -m braincode.utils $HOME_DIR 2
