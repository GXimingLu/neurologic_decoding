#!/usr/bin/env bash

DATA_DIR='../dataset'
DEVICES=$1
SPLIT=$2
OUTPUT_FILE=$3

# run decoding
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py --model_name 'gpt2-large' \
  --input_path ${DATA_DIR}/clean/init/commongen.${SPLIT}.init.txt --output_file ${OUTPUT_FILE} \
  --batch_size 32 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2