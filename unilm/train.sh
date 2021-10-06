#!/usr/bin/env bash

DATA_DIR='../dataset'
OUTPUT_DIR='finetune_model/unilm'
MODEL_RECOVER_PATH='../tmp/unilmv1-large-cased.bin'
LOG_DIR=${OUTPUT_DIR}/bert_log

CUDA_VISIBLE_DEVICES=4 python train_seq2seq.py --do_train --num_workers 16 \
  --model_type 'unilm' --model_name_or_path 'unilm-large-cased' \
  --data_dir ${DATA_DIR} \
  --train_src_file commongen.train.src_alpha.txt \
  --train_tgt_file commongen.train.tgt.txt \
  --val_src_file commongen.dev.src_alpha.txt \
  --val_tgt_file commongen.dev.tgt.txt \
  --output_dir ${OUTPUT_DIR} \
  --log_dir ${OUTPUT_DIR}/log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 64 --max_position_embeddings 64 \
  --always_truncate_tail --max_len_a 64 --max_len_b 64 \
  --mask_prob 0.7 --max_pred 20 \
  --train_batch_size 64 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 15 --subset 0.8