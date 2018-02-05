#!/bin/bash

BASE=/home/loehrj/loehrj_research/workspace/lstm
DATA=$BASE/data
MODELS=$BASE/models

SCRIPT=$BASE/lm_eval.py

MODEL=$MODELS/latest
DATA=$DATA/$1/test.txt

python $SCRIPT --model $MODEL --test-data $DATA
