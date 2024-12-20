# Library building tutorial: https://towardsdatascience.com/deep-dive-create-and-publish-your-first-python-library-f7f618719e14

# Always prefer setuptools over distutils
from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the dependencies from the requirements.txt file
with open(path.join(HERE, 'requirements.txt'), encoding='utf-8') as f:
    dependencies = f.readlines()
    dependencies = [line.strip() for line in dependencies]

# This call to setup() does all the work
setup(
    name="unihasher",
    version="0.1.3",
    description="Compare image hashes using a unified library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cp-Youfy/unihasher",
    author="Akshara Mantha, Peng Ruijia, Tan Siying",
    author_email="cpyoufy@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=["unihasher"],
    include_package_data=True,
    install_requires=dependencies
)