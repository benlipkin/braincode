# BrainCode

Project investigating human and artificial neural representations of python program comprehension and execution.

This pipeline supports three major functions.

-   **RSA** (representational similarity analysis): models program representational structure within the supported brain networks.
-   **MVPA** (multivariate pattern analysis): evaluates decoding of program features or embeddings from their respective neural representations within a collection of canonical brain networks.
-   **PRDA** (program representation decoding analysis): evaluates decoding of program features from their respective artificially-generated multivariate embeddings.

### Supported Brain Networks

-   Language
-   Multiple Demand
-   Visual
-   Auditory

### Supported Program Features

**Univariate Targets**

-   Code (code vs. sentences)
-   Content (math vs. str)
-   Structure (seq vs. for vs. if)

**Multivariate Embeddings**

-   BOW (bag of words)
-   TF-IDF (term frequency inverse document frequency)

## Installation

```bash
git clone https://github.com/benlipkin/braincode.git
cd braincode
pip install -e .
```

## Run

```bash
usage:  [-h]
        [-e {all,brain-lang,brain-MD,brain-aud,brain-vis,code-bow,code-tfidf}]
        [-f {all,test-code,task-content,task-structure,code-bow,code-tfidf}]
        {rsa,mvpa,prda}

run specified analysis type

positional arguments:
  {rsa,mvpa,prda}

optional arguments:
  -h, --help            show this help message and exit
  -e {all,brain-lang,brain-MD,brain-aud,brain-vis,code-bow,code-tfidf}, --embedding {all,brain-lang,brain-MD,brain-aud,brain-vis,code-bow,code-tfidf}
  -f {all,test-code,task-content,task-structure,code-bow,code-tfidf}, --feature {all,test-code,task-content,task-structure,code-bow,code-tfidf}
```

### RSA

**Supported embeddings**

-   brain-lang
-   brain-MD
-   brain-vis
-   brain-aud

**Sample run**

To model representational similarity of programs within the brain's Language network:

```bash
python braincode rsa -n brain-lang
```

### MVPA

**Supported embeddings**

-   brain-lang
-   brain-MD
-   brain-vis
-   brain-aud

**Supported target features**

-   test-code
-   task-content
-   task-structure
-   code-bow
-   code-tfidf

**Sample run**

To decode TF-IDF embeddings from the the brain's MD network program representations:

```bash
python braincode mvpa -n brain-MD -f code-tfidf
```

### PRDA

**Supported embeddings**

-   code-bow
-   code-tfidf

**Supported target features**

-   task-content
-   task-structure

**Sample run**

To decode program structure (seq v for v if) from the TF-IDF program representations:

```bash
python braincode prda -n code-tfidf -f task-structure
```

## Citation

If you use this work, please cite ...

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
