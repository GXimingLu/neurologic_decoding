#!/usr/bin/env bash

# path of training data
TRAIN_FILE='../dataset/s2s/commongen.train.json'
# folder used to save fine-tuned checkpoints
OUTPUT_DIR='finetune_model/unilm_v2_large'
# folder used to cache package dependencies
CACHE_DIR='tmp'


CUDA_VISIBLE_DEVICES=7 python run_seq2seq.py \
  --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR} \
  --model_type unilm --model_name_or_path unilm1-large-cased --cache_dir ${CACHE_DIR} \
  --max_source_seq_length 64 --max_target_seq_length 64 \
  --per_gpu_train_batch_size 64 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_training_epochs 15