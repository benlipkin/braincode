#!/usr/bin/env bash

set -e

conda install -yc anaconda wget
conda install -yc conda-forge unzip

HOME_DIR=$1/braincode/
cd $HOME_DIR

ZIP_NAME=inputs.zip
wget -O $ZIP_NAME https://www.dropbox.com/sh/0k5uqd33l9qcwkr/AAC1sOizqVcI5uUI2rAq0pOJa?dl=0
unzip $ZIP_NAME -d inputs