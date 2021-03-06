#!/usr/bin/env bash

KEEP_OLD="$1"

SERVER='pe'
SYNC_SERVER=""
RSYNC_EXCLUDE="--exclude predictions/ --exclude classifier/ --exclude _*"
RSYNC_EXCLUDE="--exclude classifier --exclude _*"

cd data/results

LAST_FOLDER="$(find . -maxdepth 1 -type d -name '201*' | sort -r | head -n1 | cut -c3-)/"

FOLDER_NAME="$(date "+%Y-%m-%d_%H-%M")"

# If not KEEP_OLD, rename old folder and sync
if [ -z "$KEEP_OLD" ]; then
    echo -e "########### INFO: using old folder, renaming\t($LAST_FOLDER -> $FOLDER_NAME)"
    if [ ! "$LAST_FOLDER" = "$FOLDER_NAME/" ]; then
        mv "$LAST_FOLDER" "$FOLDER_NAME" || true
    fi
# If KEEP_OLD, keep the old folder and create a new one
else
    echo -e "########### INFO: using new folder, copying\t$LAST_FOLDER -> $FOLDER_NAME"
    cp -R "$LAST_FOLDER" "$FOLDER_NAME"
fi

if [ ! -z "$SYNC_SERVER" ]; then
    echo -e "########### INFO: rsyncing remote server\t(milhouse -> ralph)"
    GET_PREDICTIONS="YES"
    ssh $SERVER sh get_results.sh $GET_PREDICTIONS #> /dev/null 2>&1
fi

if [ -z "$KEEP_OLD" ]; then
    DELETE_OPTIONS=''
else
    DELETE_OPTIONS='--delete'
fi

#DELETE_OPTIONS='--delete'
echo -e "########### INFO: rsyncing results\t\t($SERVER -> this host)"
set -x
rsync $RSYNC_EXCLUDE -avP $SERVER:bachelor-thesis/code/data/results/ $FOLDER_NAME/ $DELETE_OPTIONS
