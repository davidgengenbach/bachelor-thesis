#!/usr/bin/env bash

cd ~/bachelor-thesis/code
nohup python -u script_run_classification.py "$@" > ~/logs/script_$(date "+%Y-%m-%d_%H-%M-%S").log 2>&1 &
