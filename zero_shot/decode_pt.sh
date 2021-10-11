#!/usr/bin/env bash

export PYTHONPATH=/home/ximinglu/camera_ready

DATA_DIR='../dataset'
SPLIT='test'
OUTPUT_FILE=$1

# gpt2
CUDA_VISIBLE_DEVICES=4 python decode_gpt2.py --model_name 'gpt2-large' \
  --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/clean/constraint/${SPLIT}.constraint.json \
  --input_path ${DATA_DIR}/clean/init/commongen.${SPLIT}.init2.txt \
  --batch_size 8 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --lambda_1 1.25 --sat_tolerance 2