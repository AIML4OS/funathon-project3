#! /bin/sh

## Required by gdal
sudo apt-get update
sudo apt-get install python-dev-is-python3

## Download required packages
uv sync --frozen
