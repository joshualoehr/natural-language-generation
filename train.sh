#!/bin/bash
# Runs the training script and generates a trained model at ./models/latest

BASE=/home/loehrj/loehrj_research/workspace/lstm
DATA=$BASE/data
MODELS=$BASE/models

SCRIPT=$BASE/lm_train.py

INPUT=$DATA/$1/train.txt
OUTPUT=$MODELS/$(date -Isec | cut -c1-19)
ITERS=20000
NINPUT=5
DISP=2500

mkdir $OUTPUT
python $SCRIPT --input-file $INPUT --model-file $OUTPUT/model.ckpt --training-iters $ITERS --n-input $NINPUT --display-step $DISP &&
    rm $MODELS/latest 2> /dev/null; ln -s $OUTPUT $MODELS/latest
