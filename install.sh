#!/usr/bin/env bash


cd `dirname $0`


pip install -r requirements.txt

# install torch
pip install torch==1.2.0
pip install torchvision==0.4.0

# install package
pip install -e .
