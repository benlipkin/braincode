#!/usr/bin/env bash

set -e

conda install -yc anaconda wget
conda install -yc conda-forge tar
conda install -yc ostrokach gzip

NAME=code_transformer/
CACHE_DIR=$(pwd)/../braincode/outputs/cache/
DATASET_DIR=$CACHE_DIR/datasets/$NAME
MODEL_DIR=$CACHE_DIR/models/$NAME
BINARY_DIR=$CACHE_DIR/bin/$NAME

mkdir -p $DATASET_DIR
mkdir -p $MODEL_DIR
mkdir -p $BINARY_DIR

cd $DATASET_DIR
wget -O python.tar.gz https://syncandshare.lrz.de/dl/fi5NDSSUYPnEQ2D6zga4XtN5/python.tar.gz
wget -O multi-language.tar.gz https://syncandshare.lrz.de/dl/fiLNKYzUmYnSCtTdhVPwEyfz/multi-language.tar.gz
wget -O java-small.tar.gz https://syncandshare.lrz.de/dl/fi9phA15Ga1jHGxbp6tbWZG9/java-small.tar.gz
tar -xvzf python.tar.gz
tar -xvzf multi-language.tar.gz
tar -xvzf java-small.tar.gz

cd $MODEL_DIR
wget -O csn-single-language-models.tar.gz https://syncandshare.lrz.de/dl/fiKKgjvrkCwR3tVd5Gtu9Xpw/csn-single-language-models.tar.gz
wget -O csn-multi-language-models.tar.gz https://syncandshare.lrz.de/dl/fiRzRDTxZCKnpsiCRGaAwbiT/csn-multi-language-models.tar.gz
wget -O code2seq-models.tar.gz https://syncandshare.lrz.de/dl/fi9FdtymVXyM79rit36cDejn/code2seq-models.tar.gz
tar -xvzf csn-single-language-models.tar.gz
tar -xvzf csn-multi-language-models.tar.gz
tar -xvzf code2seq-models.tar.gz

cd $BINARY_DIR
wget -O semantic.tar.gz https://syncandshare.lrz.de/dl/fiK3DkYhvPaS1sENaGuABvi8/semantic.tar.gz
tar -xvzf semantic.tar.gz

cd $CACHE_DIR
gunzip -r .
