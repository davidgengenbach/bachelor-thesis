#!/usr/bin/env sh

TEST_PATTERN="$1"

if [ ! -z "$TEST_PATTERN" ]; then
    TEST_PATTERN="-p $TEST_PATTERN"
fi

python -m unittest discover tests $TEST_PATTERN