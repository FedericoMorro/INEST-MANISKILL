#!/bin/bash

# input parsing
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <c(ompress)/d(ecompress)> <folder_to_(de)compress>"
    exit 1
fi

ACTION=$1
FOLDER=$2

PARENT_DIR=$(dirname "$FOLDER")
BASENAME=$(basename "$FOLDER")
COMPRESSED_FILE="$FOLDER.tar.gz"

# perform (de)compression
if [ "$ACTION" == "c" ]; then
    echo "Compressing folder '$FOLDER' to '$COMPRESSED_FILE'..."
    tar -czf "$COMPRESSED_FILE" -C "$PARENT_DIR" "$BASENAME"
    echo "Compression complete."
elif [ "$ACTION" == "d" ]; then
    if [ ! -f "$COMPRESSED_FILE" ]; then
        echo "Error: Compressed file '$COMPRESSED_FILE' not found."
        exit 1
    fi
    echo "Decompressing file '$COMPRESSED_FILE' to '$PARENT_DIR'..."
    tar -xzf "$COMPRESSED_FILE" -C "$PARENT_DIR"
    echo "Decompression complete."
else
    echo "Invalid action: $ACTION. Use 'c' for compress or 'd' for decompress."
    exit 1
fi