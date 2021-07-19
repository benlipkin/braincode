#!/usr/bin/env bash

set -e

conda install -yc anaconda wget
conda install -yc conda-forge unzip

HOME_DIR=$(pwd)/../braincode/
cd $HOME_DIR

wget -O inputs.zip https://www.dropbox.com/sh/6w5s9ei928pnzod/AACRKpS58oWu1HonuaDQV3Baa
unzip inputs.zip -d inputs

