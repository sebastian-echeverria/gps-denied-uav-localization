#!/bin/bash
# Set up of pip related packages.
pip install -r requirements.txt --default-timeout=100
pip install gdal=="`gdal-config --version`.*"
