#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_PATH="isles-24-a-real-world-longitud-doi-10.5281-zenodo.16970401/train.7z"
OUTPUT_DIR="/home/renku/work/isles24/data"

echo "Extracting archive..."
mkdir -p "$OUTPUT_DIR"
7z x "$ARCHIVE_PATH" -o"$OUTPUT_DIR" -mmt=on
echo "Archive extracted at $OUTPUT_DIR"
