# BrainCode

Project investigating neural representations of python program comprehension and execution.

This pipeline supports two major functions.

-   **MVPA** (multivariate pattern analysis): evaluates decodability of program embeddings to their respective neural representations within a collection of canonical brain networks.
-   **RSA** (representational similarity analysis): models program representational structure within the supported brain networks.

### Supported Brain Networks

-   Language
-   Multiple Demand
-   Visual
-   Auditory

### Supported Program Features

#### Univariate Classification Tasks

-   Code (code vs. sentences)
-   Content (math vs. str)
-   Structure (seq vs. for vs. if)

#### Multivariate Ranked Regression Tasks

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
usage:  [-h] [-n {all,lang,MD,aud,vis}]
        [-f {all,code,content,structure,bow,tfidf}]
        {rsa,mvpa}
```

Sample MVPA: To decode TF-IDF embeddings from the MD network program representations:

```bash
python braincode mvpa -n MD -f tfidf
```

Sample RSA: To model representational similarity of programs within Language network:

```bash
python braincode rsa -n lang
```

## Citation

If you use this work, please cite ...

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
