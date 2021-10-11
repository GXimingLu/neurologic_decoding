#!/usr/bin/env bash

DEVICES=$1
SPLIT=$2
MODEL_DIR=$3
OUTPUT_DIR=$4
OUTPUT_FILE=$5

INPUT="../dataset/clean/commongen.${SPLIT}.src_alpha.txt"
TARGET="../dataset/clean/commongen.${SPLIT}.tgt.txt"

MODEL_RECOVER_PATH="${MODEL_DIR}/best_tfmr"

# bart
CUDA_VISIBLE_DEVICES=${DEVICES} python run_eval.py \
  --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${INPUT} --reference_path ${TARGET} \
  --min_tgt_length 5 --max_tgt_length 32 \
  --bs 64 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
  --save_path "${OUTPUT_DIR}/${OUTPUT_FILE}" --score_path "${OUTPUT_DIR}/score.json"

# t5-large
CUDA_VISIBLE_DEVICES=${DEVICES} python run_eval.py \
  --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${INPUT} --reference_path ${TARGET} \
  --min_tgt_length 5 --max_tgt_length 32 \
  --bs 32 --beam_size 20 --length_penalty 0.2 --ngram_size 3 \
  --save_path "${OUTPUT_DIR}/${OUTPUT_FILE}" --score_path "${OUTPUT_DIR}/score.json"