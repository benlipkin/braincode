#!/usr/bin/env bash
bash _setup_inputs.sh "$(dirname $(pwd))"
bash _setup_code_seq2seq.sh "$(dirname $(pwd))" True True
bash _setup_code_transformer.sh "$(dirname $(pwd))"
