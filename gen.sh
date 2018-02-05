#!/bin/bash

BASE=/home/loehrj/loehrj_research/workspace/lstm
DATA=$BASE/data
MODELS=$BASE/models

SCRIPT=$BASE/lm_generate.py

INPUT=$MODELS/latest
NUM=32

python $SCRIPT --model $INPUT --num-generate $NUM
