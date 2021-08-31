#!/usr/bin/env bash
##################################
# $1 can be  "$(dirname $(pwd))"
##################################
bash _setup_inputs.sh $1 
bash _setup_code_seq2seq.sh $1 False True True True True False
bash _setup_code_transformer.sh $1
