#!/usr/bin/env bash

DATA_DIR='comGen_data'
OUTPUT_DIR_NAME='finetune_model/t5-large'
CURRENT_DIR=${PWD}
OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
CACHE_DIR='tmp'

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and testing_utils.py
CUDA_VISIBLE_DEVICES=3 python finetune.py --num_workers 16 \
  --data_dir ${DATA_DIR} --output_dir=${OUTPUT_DIR} \
  --model_name_or_path t5-large --cache_dir ${CACHE_DIR} \
  --max_source_length 64 --max_target_length 64 \
  --val_max_target_length 64 --test_max_target_length 32 \
  --train_batch_size 64 --eval_batch_size 64 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 15 --do_train --do_predict --n_val=-1 \
  --gpus 1