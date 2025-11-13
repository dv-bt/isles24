#!/bin/bash
#
# Download data to session storage
mkdir data-local/
rsync -avz --progress data/ data-local/
