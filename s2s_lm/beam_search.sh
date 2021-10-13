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

CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
  --model_type 'unilm' --tokenizer_name 'unilm-large-cased' \
  --input_file ${INPUT_JSON} --split ${SPLIT} \
  --model_path ${MODEL_RECOVER_PATH} --config_path ${CONFIG_PATH} \
  --max_seq_length 64 --max_tgt_length 32 \
  --batch_size 64 --beam_size 20 --length_penalty 0.6 \
  --forbid_duplicate_ngrams --forbid_ignore_word "." \
  --output_file ${OUTPUT_FILE}

mv ${OUTPUT_FILE} ${OUTPUT_DIR}/


#CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
#  --model_type 'bert' --tokenizer_name 'bert-large-cased' \
#  --input_file ${INPUT_JSON} --split ${SPLIT} \
#  --model_path ${MODEL_RECOVER_PATH} --config_path ${CONFIG_PATH} \
#  --max_seq_length 64 --max_tgt_length 32 \
#  --batch_size 64 --beam_size 20 --length_penalty 0.6 \
#  --forbid_duplicate_ngrams --forbid_ignore_word "." \
#  --output_file ${OUTPUT_FILE}
#
#mv ${OUTPUT_FILE} ${OUTPUT_DIR}/