#!/usr/bin/env bash

# input file that you would like to decode
INPUT_JSON='../dataset/s2s/commongen.clean.test.json'
STEP=$1
SPLIT=$2
MODEL_DIR=$3
OUTPUT_DIR=$4
OUTPUT_FILE=$5

MODEL_RECOVER_PATH="${MODEL_DIR}/ckpt-${STEP}"
CONFIG_PATH="${MODEL_DIR}/ckpt-${STEP}/config.json"

CUDA_VISIBLE_DEVICES=7 python decode_seq2seq.py \
  --model_type 'unilm' --tokenizer_name 'unilm-large-cased' \
  --input_file ${INPUT_JSON} --split ${SPLIT} \
  --constraint_file ../dataset/clean/constraint/${SPLIT}.constraint.json \
  --model_path ${MODEL_RECOVER_PATH} --config_path ${CONFIG_PATH} \
  --max_seq_length 64 --max_tgt_length 32 \
  --batch_size 64 --beam_size 20 --length_penalty 0.6 \
  --forbid_duplicate_ngrams --forbid_ignore_word "." \
  --prune_factor 2 --sat_tolerance 2 \
  --output_file ${OUTPUT_FILE}

mv ${OUTPUT_FILE} decoded_sentences/${OUTPUT_DIR}/


#CUDA_VISIBLE_DEVICES=3 python decode_seq2seq.py \
#  --model_type 'bert' --tokenizer_name 'bert-large-cased' \
#  --input_file ${INPUT_JSON} --split ${SPLIT} \
#  --constraint_file ../dataset/clean/constraint/${SPLIT}.constraint.json \
#  --model_path ${MODEL_RECOVER_PATH} --config_path ${CONFIG_PATH} \
#  --max_seq_length 64 --max_tgt_length 32 \
#  --batch_size 64 --beam_size 20 --length_penalty 0.6 \
#  --forbid_duplicate_ngrams --forbid_ignore_word "." \
#  --prune_factor 2 --sat_tolerance 2 \
#  --output_file ${OUTPUT_FILE}
#
#mv ${OUTPUT_FILE} decoded_sentences/${OUTPUT_DIR}/