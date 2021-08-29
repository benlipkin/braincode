from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy==1.18.1",
    "scipy==1.4.1",
    "scikit_learn==0.24.1",
    "torch==1.4.0",
    "tensorflow==2.3.0",
    "transformers==3.1.0",
    "datasets==1.9.0",
    "matplotlib==3.3.4",
    "joblib==0.14.1",
    "tqdm==4.43.0",
    "code-transformer @ git+https://github.com/danielzuegner/code-transformer.git",
    "torchtext==0.10",
    "dill",
    "astor",
]

setup(
    name="BrainCode",
    version="0.1.0",
    description="an investigation of computer program representations.",
    long_description=readme,
    url="https://github.com/benlipkin/braincode",
    author="Benjamin Lipkin",
    author_email="lipkinb@mit.edu",
    license="MIT",
    packages=find_packages(where="braincode"),
    install_requires=requirements,
    python_requires=">=3.7",
)
