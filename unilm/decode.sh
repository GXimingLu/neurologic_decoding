#!/usr/bin/env bash

export PYTHONPATH=/home/ximingl/comGen

DATA_DIR='../dataset/clean'
EPOCH=$1
EVAL_SPLIT=$2
MODEL_DIR=$3
OUTPUT_DIR=$4
OUTPUT_FILE=$5

MODEL_RECOVER_PATH="${MODEL_DIR}/model.${EPOCH}.bin"

# run decoding
CUDA_VISIBLE_DEVICES=1 python decode_seq2seq.py \
  --model_type 'unilm' --model_name_or_path 'unilm-large-cased' \
  --input_file ${DATA_DIR}/commongen.${EVAL_SPLIT}.src_alpha.txt --split ${EVAL_SPLIT} \
  --constraint_file ${DATA_DIR}/constraint/${EVAL_SPLIT}.constraint.json \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 64 --max_tgt_length 32 \
  --batch_size 64 --beam_size 20 --length_penalty 0.6 \
  --forbid_duplicate_ngrams --forbid_ignore_word "." \
  --prune_factor 50 --sat_tolerance 2 \
  --output_file ${OUTPUT_FILE}

mv ${OUTPUT_FILE} reproduce/${OUTPUT_DIR}/