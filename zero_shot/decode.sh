#!/usr/bin/env bash

export PYTHONPATH=/home/ximinglu/open_source

DATA_DIR='../dataset'
SPLIT='test'

DEVICES=$1
FACTOR=$2
OUTPUT_FILE='generation/raw/test'

# gpt2
CUDA_VISIBLE_DEVICES=${DEVICES} python decode_gpt2.py --model_name 'gpt2-large' \
  --output_file ${OUTPUT_FILE}.${FACTOR} \
  --constraint_file ${DATA_DIR}/clean/constraint/${SPLIT}.constraint.json.${FACTOR} \
  --batch_size 32 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --lambda_1 1.25 --sat_tolerance 2