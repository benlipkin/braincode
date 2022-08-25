#!/usr/bin/env bash
##################################
# $1 is base_dir
##################################
base_dir=${1:-$(dirname $(pwd))}
bash _setup_inputs.sh $base_dir
bash _setup_benchmarks.sh $base_dir