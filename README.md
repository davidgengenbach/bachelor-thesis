# Text Classification using Concept Maps

[Protocol](other/protocol.md)

## Instructions

Developed and tested with Python 3.6. Recommendation: [anaconda](https://www.anaconda.com/download)

```bash
git clone --depth 1 git@github.com:davidgengenbach/bachelor-thesis.git

cd code

# Bootstrap once
./script_bootstrap.sh

# Generate co-occurrence graphs
./script_create_coocurrence_graphs.py

# To run the classifications. See below for more information
./script_run_classification.py
```

Please see the individual scripts for parameters.

The concept maps have to be downloaded seperately (currently).

## To Do
- Provide concept maps as download