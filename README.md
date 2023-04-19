
[![Tests](https://github.com/benlipkin/braincode/actions/workflows/testing.yml/badge.svg)](https://github.com/benlipkin/braincode/actions/workflows/testing.yml)

# BrainCode

Project investigating human and artificial neural representations of code.

This branch is currently under development, and should be considered unstable. To replicate specific papers, `git checkout` the corresponding branch, e.g., `NeurIPS2022`, and follow instructions in the `README.md`.


This pipeline supports several major functions.

-   **MVPA** (multivariate pattern analysis) evaluates decoding of **code properties** or **code model** representations from their respective **brain representations** within a collection of canonical **brain regions**.
-   **RSA** (representational similarity analysis) is also supported as an alternative to MVPA.
-   **VWEA** (voxel-wise encoding analysis) evaluates prediction of voxel-level activation patterns using **code properties** and **code model** representations as features.
-   **NLEA** (network-level encoding analysis) uses the same features to evaluate encoding of mean network-level activation strength.
-   **PRDA** (program representation decoding analysis) evaluates decoding of **code properties** from **code model** representations.
-   **PREA** (program representation encoding analysis) evaluates encoding of **code model** representations using the set of **code properties** explored in this work.

_Note: **VWEA** and **NLEA** also support ceiling estimates at the network level, calculated via an identical pipeline but with the features being the representations of other participants to the same stimuli rather than the properties extracted from those stimuli. To invoke a ceiling analysis, prefix the requested analysis type with a "C", e.g., **CNLEA**._

### Supported Brain Regions

-   `brain-md_lh` (Multiple Demand Network: Left Hemisphere)
-   `brain-md_rh` (Multiple Demand Network: Right Hemisphere)
-   `brain-lang_lh` (Language Network: Left Hemisphere)
-   `brain-lang_rh` (Language Network: Right Hemisphere)

### Supported Code Features

**Code Properties**

-   `task-structure` (seq vs. for vs. if) <sup>\*ControlFlow</sup>
-   `task-content` (math vs. str) <sup>\*DataType</sup>
-   `task-nodes` (# of nodes in AST) <sup>\*ASTNodes</sup>
-   `task-lines` (# of runtime steps during execution) <sup>\*LinesExecuted</sup>

**Code Models**

Baseline:

-   `code-tokens` (arbitrary projection encoding presence of individual tokens)

LLM Suite (CodeGen<sup>[1](https://arxiv.org/pdf/2203.13474.pdf)</sup>):

-   `code-llm_350m_nl`
-   `code-llm_2b_nl`
-   `code-llm_6b_nl`
-   `code-llm_16b_nl`
-   `code-llm_350m_mono`
-   `code-llm_2b_mono`
-   `code-llm_6b_mono`
-   `code-llm_16b_mono`

_Note: checkpoints vary in size and pre-training (`nl`—ThePile; `mono`—ThePile+BigQuery+BigPython)_
## Installation

Requirements: [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), [GNU Make](https://www.gnu.org/software/make/manual/make.html)

```bash
git clone --branch main --depth 1 https://github.com/benlipkin/braincode
cd braincode
make setup
```

## Run

```bash
usage: __main__.py [-h] [-f FEATURE] [-t TARGET] [-m METRIC] [-d CODE_MODEL_DIM] [-p BASE_PATH] [-s] [-b] {mvpa,rsa,vwea,nlea,cvwea,cnlea,prda,prea}

run specified analysis type

positional arguments:
  {mvpa,rsa,vwea,nlea,cvwea,cnlea,prda,prea}

optional arguments:
  -h, --help            show this help message and exit
  -f FEATURE, --feature FEATURE
  -t TARGET, --target TARGET
  -m METRIC, --metric METRIC
  -d CODE_MODEL_DIM, --code_model_dim CODE_MODEL_DIM
  -p BASE_PATH, --base_path BASE_PATH
  -s, --score_only
  -b, --debug
```

_Note: BASE_PATH must be specified to match setup.sh if changed from default._

**Sample calls**

```bash
# basic examples
python -m braincode mvpa -f brain-md_lh -t task-structure # brain -> {task, model}
python -m braincode rsa -f brain-lang_lh -t code-llm_2b_nl # brain <-> {task, model}
python -m braincode vwea -f brain-md_rh -t code-tokens # brain <- {task, model}
python -m braincode nlea -f brain-lang_rh -t task-content # brain <- {task, model}
python -m braincode prda -f code-llm_350m_mono -t task-lines # model -> task
python -m braincode prea -f code-tokens -f task-content # model <- task

# more complex examples
python -m braincode cnlea -f all -m SpearmanRho --score_only # check metrics module for all options
python -m braincode mvpa -f brain-lang_lh+brain-lang_rh -t code-tokens -d 64 -p $BASE_PATH
python -m braincode vwea -t task-content+task-structure+task-nodes+task-lines
# note how `+` operator can be used to join multiple representations via concatenation
```

## Citation

If you use this work, please cite:
```bibtex
@inproceedings{SrikantLipkin2022,
	title={Convergent Representations of Computer Programs in Human and Artificial Neural Networks},
	author={Shashank Srikant* and Ben Lipkin* and Anna A Ivanova and Evelina Fedorenko and Una-May O'Reilly},
	booktitle={Advances in Neural Information Processing Systems},
	editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
	year={2022},
	url={https://openreview.net/forum?id=AqexjBWRQFx}
}
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
