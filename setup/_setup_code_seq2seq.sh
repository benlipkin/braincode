#!/usr/bin/env bash

# $1: Base path to store data, models
# $2: Download codenet-python dataset: True, False
# $3: Train codenet-python dataset: True, False

set -e

# conda install -yc anaconda wget
# conda install -yc conda-forge tar

NAME=code_seq2seq
CACHE_DIR=$1/braincode/.cache
DATASET_DIR=$CACHE_DIR/datasets/$NAME
MODEL_DIR=$CACHE_DIR/models/$NAME
BINARY_DIR=$CACHE_DIR/bin/$NAME
LOG_DIR=$CACHE_DIR/logs/$NAME
ENV_DIR=$HOME/.config/$NAME

mkdir -p $DATASET_DIR
mkdir -p $MODEL_DIR
mkdir -p $BINARY_DIR
mkdir -p $LOG_DIR
mkdir -p $ENV_DIR

TAR_NAME=codenet-python
if [ "$2" == "True" ]; then
    cd $DATASET_DIR
    wget -O $TAR_NAME.tar.gz https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_Python800.tar.gz?_ga=2.140558578.1761838737.1630155855-230520885.1625251724
    mkdir $TAR_NAME
    tar -xvzf $TAR_NAME.tar.gz -C $TAR_NAME --strip-components 1
fi

if [ "$3" == "True" ]; then
    TRAIN_FILES_NAME="train_files.txt"
    TEST_FILES_NAME="test_files.txt"
    python -m seq2seq.split_data $DATASET_DIR/$TAR_NAME $DATASET_DIR/$TRAIN_FILES_NAME $DATASET_DIR/$TEST_FILES_NAME .py 
    python -m seq2seq.tokenize $DATASET_DIR/$TAR_NAME $DATASET_DIR/$TRAIN_FILES_NAME $DATASET_DIR/$TEST_FILES_NAME
    #python -m seq2seq.train \
    #--train_path /data/seq2seq/input/tokenized/train.txt \
    #--dev_path /data/seq2seq/input/tokenized/test.txt \
    #--obf_path /data/seq2seq/input/tokenized/obf.txt\
    #--expt_dir /data/seq2seq/experiments/\
    #--reps_store_path /data/malware_project/processed/5k/reps_5k/
fi

#cd $ENV_DIR
#echo "
#export CODE_SEQ2SEQ_DATA_PATH=$DATASET_DIR
#export CODE_SEQ2SEQ_MODELS_PATH=$MODEL_DIR
#export CODE_SEQ2SEQ_LOGS_PATH=$LOG_DIR
#" > $ENV_DIR/.env
