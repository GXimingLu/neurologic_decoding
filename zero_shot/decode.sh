#!/usr/bin/env bash

DATA_DIR='../dataset'

DEVICES=$1
SPLIT=$2
OUTPUT_FILE=$3

# gpt2
CUDA_VISIBLE_DEVICES=${DEVICES} python decode_gpt2.py --model_name 'gpt2-large' \
  --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/clean/constraint/${SPLIT}.constraint.json \
  --batch_size 32 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 --beta 1.25 --early_stop 10