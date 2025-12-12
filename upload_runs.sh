#!/bin/bash
#
# Upload runs to remote storage
mkdir data/runs/
rsync -avz --progress auto3dseg-runs/ data/runs/
