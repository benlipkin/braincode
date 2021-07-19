#!/usr/bin/env bash

set -e

conda install -yc anaconda wget
conda install -yc conda-forge tar

NAME=code_transformer/
CACHE_DIR=$(pwd)/../braincode/ouptuts/cache/
DATASET_DIR=$CACHE_DIR/datasets/$NAME
MODEL_DIR=$CACHE_DIR/models/$NAME

mkdir -p DATASET_DIR
mkdir -p MODEL_DIR
