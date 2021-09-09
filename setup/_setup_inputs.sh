#!/usr/bin/env bash

set -e

conda install -yc anaconda wget
conda install -yc conda-forge unzip

HOME_DIR=$1/braincode/
cd $HOME_DIR

wget -O inputs.zip https://www.dropbox.com/s/78e0r8kmmnwfa1m/inputs.zip?dl=0
unzip inputs.zip -d inputs
