from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as reqs_file:
    requirements = reqs_file.read().split("\n")

setup(
    name="braincode",
    version="0.2",
    description="an investigation of computer program representations.",
    long_description=readme,
    author="Benjamin Lipkin",
    author_email="lipkinb@mit.edu",
    license="MIT",
    packages=find_packages(where="braincode"),
    install_requires=requirements,
    python_requires=">=3.9",
)
