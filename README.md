# BrainCode

Project investigating human and artificial neural representations of python program comprehension and execution.

This pipeline supports three major functions.

-   **RSA** (representational similarity analysis): models program representational structure within the supported brain networks.
-   **MVPA** (multivariate pattern analysis): evaluates decoding of program benchmark tasks or embeddings from their respective neural representations within a collection of canonical brain networks.
-   **PRDA** (program representation decoding analysis): evaluates decoding of program benchmark tasks from their respective in-silico embeddings.

### Supported Brain Networks

-   Language
-   Multiple Demand (MD)
-   Visual
-   Auditory

### Supported Program Features

**Benchmark Tasks**

-   Code (code vs. sentences)
-   Content (math vs. str)
-   Language (english vs. japanese)
-   Structure (seq vs. for vs. if)

**Program Embeddings**

-   Bag Of Words
-   TF-IDF
-   XLNet<sup> [1](https://arxiv.org/pdf/1906.08237.pdf)</sup>
-   CodeTransformer<sup> [2](https://arxiv.org/pdf/2103.11318.pdf)</sup>
-   CodeBERTa<sup> [3](https://huggingface.co/huggingface/CodeBERTa-small-v1)</sup>

## Installation

Requirements: [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda create -n braincode python=3.7
source activate braincode
git clone --depth 1 https://github.com/benlipkin/braincode.git
cd braincode
pip install . # -e for development mode
cd setup
source setup.sh # downloads 'large' files, e.g. datasets, models
```

## Run

```bash
usage:  [-h]
        [-f {all,brain-lang,brain-MD,brain-aud,brain-vis,code-bow,code-tfidf,code-xlnet,code-ct,code-codeberta}]
        [-t {all,test-code,task-content,task-lang,task-structure,code-bow,code-tfidf,code-xlnet,code-ct,code-codeberta}]
        {rsa,mvpa,prda}

run specified analysis type

positional arguments:
  {rsa,mvpa,prda}

optional arguments:
  -h, --help            show this help message and exit
  -f {all,brain-lang,brain-MD,brain-aud,brain-vis,code-bow,code-tfidf,code-xlnet,code-ct,code-codeberta}, --feature {all,brain-lang,brain-MD,brain-aud,brain-vis,code-bow,code-tfidf,code-xlnet,code-ct,code-codeberta}
  -t {all,test-code,task-content,task-lang,task-structure,code-bow,code-tfidf,code-xlnet,code-ct,code-codeberta}, --target {all,test-code,task-content,task-lang,task-structure,code-bow,code-tfidf,code-xlnet,code-ct,code-codeberta}
```

### RSA

**Supported features**

-   brain-lang
-   brain-MD
-   brain-vis
-   brain-aud

**Sample run**

To model representational similarity of programs within the brain's Language network:

```bash
python braincode rsa -f brain-lang
```

### MVPA

**Supported features**

-   brain-lang
-   brain-MD
-   brain-vis
-   brain-aud

**Supported targets**

-   test-code
-   task-content
-   task-lang
-   task-structure
-   code-bow
-   code-tfidf
-   code-xlnet
-   code-ct
-   code-codeberta

**Sample run**

To decode TF-IDF embeddings from the brain's MD network program representations:

```bash
python braincode mvpa -f brain-MD -t code-tfidf
```

### PRDA

**Supported features**

-   code-bow
-   code-tfidf
-   code-xlnet
-   code-ct
-   code-codeberta

**Supported targets**

-   task-content
-   task-lang
-   task-structure

**Sample run**

To decode program structure (seq vs. for vs. if) from the CodeBERTa program representations:

```bash
python braincode prda -f code-codeberta -t task-structure
```

## Citation

If you use this work, please cite ...

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
