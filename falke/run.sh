#!/usr/bin/env bash


docker build -t falke .
docker run -v $(pwd)/code:/code -it falke bash
