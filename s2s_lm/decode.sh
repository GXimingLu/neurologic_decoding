#!/usr/bin/env bash

# input file that you would like to decode
DEVICES=$1
STEP=$2
SPLIT=$3
MODEL_DIR=$4
OUTPUT_DIR=$5
OUTPUT_FILE=$6

INPUT_JSON="../dataset/s2s/commongen.clean.${SPLIT}.json"

MODEL_RECOVER_PATH="${MODEL_DIR}/ckpt-${STEP}"
CONFIG_PATH="${MODEL_DIR}/ckpt-${STEP}/config.json"

# unilm v2
CUDA_VISIBLE_DEVICES=${DEVICES} python decode_seq2seq.py \
  --model_type 'unilm' --tokenizer_name 'unilm-large-cased' \
  --input_file ${INPUT_JSON} --split ${SPLIT} \
  --constraint_file ../dataset/clean/constraint/${SPLIT}.constraint.json \
  --model_path ${MODEL_RECOVER_PATH} --config_path ${CONFIG_PATH} \
  --max_seq_length 64 --max_tgt_length 32 \
  --batch_size 64 --beam_size 20 --length_penalty 0.6 \
  --forbid_duplicate_ngrams --forbid_ignore_word "." \
  --prune_factor 50 --sat_tolerance 2 --beta 0 --early_stop 1.5 \
  --output_file ${OUTPUT_FILE}

mv ${OUTPUT_FILE} ${OUTPUT_DIR}/

# bert
#CUDA_VISIBLE_DEVICES=${DEVICES} python decode_seq2seq.py \
#  --model_type 'bert' --tokenizer_name 'bert-large-cased' \
#  --input_file ${INPUT_JSON} --split ${SPLIT} \
#  --constraint_file ../dataset/clean/constraint/${SPLIT}.constraint.json \
#  --model_path ${MODEL_RECOVER_PATH} --config_path ${CONFIG_PATH} \
#  --max_seq_length 64 --max_tgt_length 32 \
#  --batch_size 64 --beam_size 20 --length_penalty 0.6 \
#  --forbid_duplicate_ngrams --forbid_ignore_word "." \
#  --prune_factor 50 --sat_tolerance 2 --beta 0 --early_stop 1.5 \
#  --output_file ${OUTPUT_FILE}
#
#mv ${OUTPUT_FILE} ${OUTPUT_DIR}/