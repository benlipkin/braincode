#!/usr/bin/env bash

set -e

conda install -yc anaconda wget
conda install -yc conda-forge tar

NAME=code_transformer
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

cd $DATASET_DIR
wget -O python.tar.gz https://www.dropbox.com/s/ukve7lu6t9d6kfu/python.tar.gz?dl=0
tar -xvzf python.tar.gz

cd $MODEL_DIR
wget -O csn-single-language-models.tar.gz https://syncandshare.lrz.de/dl/fiKKgjvrkCwR3tVd5Gtu9Xpw/csn-single-language-models.tar.gz
tar -xvzf csn-single-language-models.tar.gz

cd $BINARY_DIR
wget -O semantic.tar.gz https://www.dropbox.com/s/vxpcjs2myi8yych/semantic.tar.gz?dl=0
tar -xvzf semantic.tar.gz

cd $ENV_DIR
echo "
export CODE_TRANSFORMER_DATA_PATH=$DATASET_DIR
export CODE_TRANSFORMER_BINARY_PATH=$BINARY_DIR
export CODE_TRANSFORMER_MODELS_PATH=$MODEL_DIR
export CODE_TRANSFORMER_LOGS_PATH=$LOG_DIR

export CODE_TRANSFORMER_CSN_RAW_DATA_PATH=$DATASET_DIR/raw/csn
export CODE_TRANSFORMER_CODE2SEQ_RAW_DATA_PATH=$DATASET_DIR/raw/code2seq
export CODE_TRANSFORMER_CODE2SEQ_EXTRACTED_METHODS_DATA_PATH=$DATASET_DIR/raw/code2seq-methods

export CODE_TRANSFORMER_DATA_PATH_STAGE_1=$DATASET_DIR/stage1
export CODE_TRANSFORMER_DATA_PATH_STAGE_2=$DATASET_DIR/stage2

export CODE_TRANSFORMER_JAVA_EXECUTABLE=java
export CODE_TRANSFORMER_JAVA_PARSER_EXECUTABLE=$BINARY_DIR/java-parser-1.0-SNAPSHOT.jar
export CODE_TRANSFORMER_JAVA_METHOD_EXTRACTOR_EXECUTABLE=$BINARY_DIR/JavaMethodExtractor-1.0.0-SNAPSHOT.jar
export CODE_TRANSFORMER_SEMANTIC_EXECUTABLE=$BINARY_DIR/semantic
" > $ENV_DIR/.env
