#!/usr/bin/env bash

KEEP_OLD="$1"

SERVER='ba'


LAST_FOLDER=$(find . -depth 1 -type d -name '2017*' | sort -r | head -n1 | cut -c3-)

FOLDER_NAME="$(date "+%Y-%m-%d_%H-%M")"

# If not KEEP_OLD, rename old folder and sync
if [ -z "$KEEP_OLD" ]; then
    echo "INFO: using old folder, renaming ($LAST_FOLDER -> $FOLDER_NAME)"
    if [ ! "$LAST_FOLDER" = "$FOLDER_NAME" ]; then
        mv "$LAST_FOLDER" "$FOLDER_NAME" || true
    fi
# If KEEP_OLD, keep the old folder and create a new one
else
    echo "INFO: using new folder, copying ($FOLDER_NAME)"
    cp -R $LAST_FOLDER $FOLDER_NAME
fi

set -x

echo "rsyncing remote server"
GET_PREDICTIONS="YES"
ssh $SERVER sh get_results.sh $GET_PREDICTIONS > /dev/null 2>&1

if [ -z "$KEEP_OLD" ]; then
    DELETE_OPTIONS=''
else
    DELETE_OPTIONS='--delete'
fi




rsync -avz $SERVER:results/ $FOLDER_NAME/ $DELETE_OPTIONS

