#!/usr/bin/env bash

DATA_DIR='../dataset/clean'
DEVICES=$1
EPOCH=$2
EVAL_SPLIT=$3
MODEL_DIR=$4
OUTPUT_DIR=$5
OUTPUT_FILE=$6

MODEL_RECOVER_PATH="${MODEL_DIR}/model.${EPOCH}.bin"

# run decoding
CUDA_VISIBLE_DEVICES=7 python beam_search.py \
  --model_type 'unilm' --model_name_or_path 'unilm-large-cased' \
  --input_file ${DATA_DIR}/commongen.${EVAL_SPLIT}.src_alpha.txt --split ${EVAL_SPLIT} \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 64 --max_tgt_length 32 \
  --batch_size 64 --beam_size 20 --length_penalty 0.6 \
  --forbid_duplicate_ngrams --forbid_ignore_word "." \
  --output_file ${OUTPUT_FILE}

mv ${OUTPUT_FILE} ${OUTPUT_DIR}/
