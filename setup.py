from setuptools import find_packages, setup

setup(
    name="BrainCode",
    version="0.1.0",
    description="",
    url="https://github.com/benlipkin/braincode",
    author="Benjamin Lipkin",
    author_email="lipkinb@mit.edu",
    license="MIT",
    packages=find_packages(where="braincode"),
    install_requires=[
        "numpy",
        "scipy",
        "sklearn",
        "pytorch",
        "tensorflow",
        "matplotlib",
        "joblib",
        "tqdm",
    ],
    python_requires=">=3.7",
)
