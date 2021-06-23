from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "scipy",
    "sklearn",
    "torch",
    "tensorflow",
    "transformers",
    "matplotlib",
    "joblib",
    "tqdm",
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
    python_requires=">=3.6",
)
