# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name="antifold",
    version="0.0.3",
    packages=find_packages(),
    description="Inverse folding of antibodies",
    url="https://github.com/Magnushhoie/antifold_web/",
    author="Magnus Haraldson Høie & Alissa Hummer",
    author_email="maghoi@dtu.dk & alissa.hummer@stcatz.ox.ac.uk",
    install_requires=REQUIREMENTS,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
    ],
)
