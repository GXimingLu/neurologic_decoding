#!/usr/bin/env bash

DATA_DIR='../dataset'
DEVICES=$1
SPLIT=$2
MODEL_RECOVER_PATH=$3
OUTPUT_FILE=$4

# run decoding
CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${DATA_DIR}/lm/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/clean/constraint/${SPLIT}.constraint.json \
  --batch_size 32 --beam_size 20 --max_tgt_length 64 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 50 --sat_tolerance 2 --beta 0 --early_stop 1.5

