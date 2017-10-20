#!/usr/bin/env bash

LAST_LOG=$(find ~/logs -maxdepth 1 -type f -name '*2017*.log' | sort -r | head -n1)

TO_GREP="$1"

if [ -e "$TO_GREP" ]; then
	LAST_LOG="$TO_GREP"
	TO_GREP=""
fi

if [ ! -z "$TO_GREP" ]; then
	tail -n99999999 -f $LAST_LOG | grep "$TO_GREP"
else
	tail -n99999999 -f $LAST_LOG
fi
