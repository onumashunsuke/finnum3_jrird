#!/bin/bash
# Copyright 2022 The Japan Research Institute, Limited.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SRC_DIR=/root/work/src
BASE_MODEL_DIR=/root/work/models/base
TUNED_MODEL_DIR=/root/work/models/tuned
TRAIN_DIR=/root/work/data/processed/ordered_split_num
DEV_DIR=/root/work/data/processed/dev
TEST_DIR=/root/work/data/processed/test
RESULT_DIR=/root/work/results

time python3 $SRC_DIR/models/digit_bert_base/run_digit.py \
    --model_name_or_path $BASE_MODEL_DIR/bert-base-cased \
    --train_file $TRAIN_DIR/preprocess_digit/fold3/train.json \
    --validation_file $TRAIN_DIR/preprocess_digit/fold3/val.json \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 20 \
    --output_dir $TUNED_MODEL_DIR/digit_bert_base_ordered_split_num/digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3 \
    --save_strategy no \
    --overwrite_cache \
    --fp16 \
    --fp16_opt_level O2

time python3 $SRC_DIR/models/digit_bert_base/run_digit.py \
  --model_name_or_path $TUNED_MODEL_DIR/digit_bert_base_ordered_split_num/digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3 \
  --train_file $TRAIN_DIR/preprocess_digit/fold3/train.json \
  --validation_file $TRAIN_DIR/preprocess_digit/fold3/val.json \
  --test_file $TRAIN_DIR/preprocess_digit/fold3/val.json \
  --do_predict \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --output_dir $RESULT_DIR/digit_bert_base_ordered_split_num/digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3 \
  --overwrite_cache

python3 $SRC_DIR/evaluate/evaluate.py \
  --prediction_file $RESULT_DIR/digit_bert_base_ordered_split_num/digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3/predict_results_None.txt \
  --data_file $TRAIN_DIR/split_num/fold3/val.json \
  --result_file $RESULT_DIR/digit_bert_base_ordered_split_num/digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3/metrics.txt \
  --claim_detect_only \
  --eval_validation \
  --merged_prediction

time python3 $SRC_DIR/models/digit_bert_base/run_digit.py \
  --model_name_or_path $TUNED_MODEL_DIR/digit_bert_base_ordered_split_num/digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3 \
  --train_file $TRAIN_DIR/preprocess_digit/fold3/train.json \
  --validation_file $TRAIN_DIR/preprocess_digit/fold3/val.json \
  --test_file $DEV_DIR/dev_preprocess_digit.json \
  --do_predict \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --output_dir $RESULT_DIR/digit_bert_base_ordered_split_num/digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3/dev \
  --overwrite_cache

python3 $SRC_DIR/evaluate/evaluate.py \
  --prediction_file $RESULT_DIR/digit_bert_base_ordered_split_num/digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3/dev/predict_results_None.txt \
  --data_file $DEV_DIR/dev_num.json \
  --result_file $RESULT_DIR/digit_bert_base_ordered_split_num/digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3/dev/metrics.txt \
  --claim_detect_only \
  --merged_prediction \
  --unmerged_data_file $DEV_DIR/dev.json

time python3 $SRC_DIR/models/digit_bert_base/run_digit.py \
  --model_name_or_path $TUNED_MODEL_DIR/digit_bert_base_ordered_split_num/digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3 \
  --train_file $TRAIN_DIR/preprocess_digit/fold3/train.json \
  --validation_file $TRAIN_DIR/preprocess_digit/fold3/val.json \
  --test_file $TEST_DIR/test_preprocess_digit.json \
  --do_predict \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --output_dir $RESULT_DIR/digit_bert_base_ordered_split_num/digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3/test \
  --overwrite_cache

