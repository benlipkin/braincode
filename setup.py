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

classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "Programming Language :: Python",
    "Natural Language :: English",
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
    classifiers=classifiers,
)
