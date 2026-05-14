#!/bin/bash

# input parsing
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <c(ompress)/d(ecompress)> <folder_to_(de)compress>"
    exit 1
fi

ACTION=$1
FOLDER=$2

PARENT_DIR=$(dirname "$FOLDER")

# perform (de)compression
if [ "$ACTION" == "c" ]; then
    BASENAME=$(basename "$FOLDER")
    COMPRESSED_FILE="$FOLDER.tar.gz"

    echo "Compressing folder '$FOLDER' to '$COMPRESSED_FILE'..."
    tar -czvf "$COMPRESSED_FILE" -C "$PARENT_DIR" "$BASENAME"
    echo "Compression complete."

elif [ "$ACTION" == "d" ]; then
    COMPRESSED_FILE="$FOLDER"

    if [ ! -f "$COMPRESSED_FILE" ]; then
        echo "Error: Compressed file '$COMPRESSED_FILE' not found."
        exit 1
    fi

    echo "Decompressing file '$COMPRESSED_FILE' to '$PARENT_DIR'..."
    tar -xzvf "$COMPRESSED_FILE" -C "$PARENT_DIR"
    echo "Decompression complete."
else
    echo "Invalid action: $ACTION. Use 'c' for compress or 'd' for decompress."
    exit 1
fi