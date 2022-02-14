#!/usr/bin/env bash
sudo rm *.so
sudo rm -r build/
python setup.py build develop
