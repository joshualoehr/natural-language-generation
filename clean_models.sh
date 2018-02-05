#!/bin/bash
# Removes all but the latest saved model

BASE=/home/loehrj/loehrj_research/workspace/lstm
MODELS=$BASE/models
LATEST=$(ls -l models/latest | rev | cut -d" " -f1 | rev)

find $MODELS -name "2018-*" | grep -v "$LATEST" | xargs -n1 -I{} rm -r {}

