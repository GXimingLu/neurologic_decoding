#!/usr/bin/env bash

DATA_DIR='../dataset/clean'
SPLIT=$1
MODEL_DIR=$2
OUTPUT_DIR=$3
OUTPUT_FILE=$4

export PYTHONPATH=/home/ximingl/comGen

INPUT="${DATA_DIR}/commongen.${SPLIT}.src_alpha.txt"
TARGET="${DATA_DIR}/commongen.${SPLIT}.tgt.txt"

MODEL_RECOVER_PATH="${MODEL_DIR}/best_tfmr"

# bart
#CUDA_VISIBLE_DEVICES=4 python decode.py \
#  --model_name ${MODEL_RECOVER_PATH} \
#  --input_path ${INPUT} --reference_path ${TARGET} \
#  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
#  --min_tgt_length 5 --max_tgt_length 32 \
#  --bs 64 --beam_size 20 --length_penalty 0.1 --ngram_size 3 \
#  --prune_factor 50 --sat_tolerance 2 \
#  --save_path "${OUTPUT_DIR}/${OUTPUT_FILE}.${SPLIT}" --score_path "${OUTPUT_DIR}/${OUTPUT_FILE}.json"

# t5-large
CUDA_VISIBLE_DEVICES=5 python decode.py \
  --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${INPUT} --reference_path ${TARGET} \
  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
  --min_tgt_length 5 --max_tgt_length 32 \
  --bs 32 --beam_size 20 --length_penalty 0.2 --ngram_size 3 \
  --prune_factor 50 --sat_tolerance 2 \
  --save_path "${OUTPUT_DIR}/${OUTPUT_FILE}.${SPLIT}" --score_path "${OUTPUT_DIR}/${OUTPUT_FILE}.json"

# t5-base
#CUDA_VISIBLE_DEVICES=6 python decode.py \
#  --model_name ${MODEL_RECOVER_PATH} \
#  --input_path ${INPUT} --reference_path ${TARGET} \
#  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
#  --min_tgt_length 5 --max_tgt_length 32 \
#  --bs 64 --beam_size 20 --length_penalty 0.2 --ngram_size 3 \
#  --prune_factor 50 --sat_tolerance 2 \
#  --save_path "${OUTPUT_DIR}/${OUTPUT_FILE}.${SPLIT}" --score_path "${OUTPUT_DIR}/${OUTPUT_FILE}.json"