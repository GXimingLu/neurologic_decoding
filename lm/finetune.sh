#!/usr/bin/env bash

MODEL_NAME=$1
DATA_DIR='../dataset/lm'
OUTPUT_DIR_NAME="finetune_model/${MODEL_NAME}"
CURRENT_DIR=${PWD}
OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
CACHE_DIR='tmp'

# Add parent directory to python path to access lightning_base.py and testing_utils.py
CUDA_VISIBLE_DEVICES=2 python finetune_gpt2.py --output_dir=${OUTPUT_DIR} \
  --train_data_file ${DATA_DIR}/train.txt --eval_data_file ${DATA_DIR}/dev.txt \
  --block_size 76 --line_by_line \
  --model_type gpt2 --model_name_or_path ${MODEL_NAME} --cache_dir ${CACHE_DIR} \
  --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_steps 900 \
  --num_train_epochs 15 --do_train --do_eval --do_predict --evaluate_during_training \
  --logging_dir ${OUTPUT_DIR}/log --logging_steps 611 --save_steps 300 --eval_steps 900