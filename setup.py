# -*- coding: utf-8 -*-
from setuptools import find_packages, setup
from pathlib import Path

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name="antifold",
    version="0.3.1",
    packages=find_packages(),
    description="Inverse folding of antibodies",
    url="https://github.com/oxpig/AntiFold/",
    author="Magnus Haraldson Høie & Alissa Hummer",
    author_email="maghoi@dtu.dk",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=REQUIREMENTS,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
    ],
)
