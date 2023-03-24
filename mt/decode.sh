#!/usr/bin/env bash

DATA_DIR='../dataset'
DEVICES=$1
OUTPUT_FILE=$2

# run decoding
CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name 'Helsinki-NLP/opus-mt-en-fr' \
  --input_file ${DATA_DIR}/mt/input.txt --constraint_file ${DATA_DIR}/mt/fr_constrain.pk --output_file ${OUTPUT_FILE} \
  --batch_size 32 --beam_size 20 --max_tgt_length 64 --min_tgt_length 5 \
  --ngram_size 5 --length_penalty 0.2 --prune_factor 1.5 --sat_tolerance 2
