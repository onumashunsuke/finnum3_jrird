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

time python3 $SRC_DIR/models/digit_bert_base_joint_learning/run_digit_joint_learning.py \
    --model_name_or_path $BASE_MODEL_DIR/bert-base-cased \
    --train_file $TRAIN_DIR/preprocess_digit_with_category/fold4/train.json \
    --validation_file $TRAIN_DIR/preprocess_digit_with_category/fold4/val.json \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --output_dir $TUNED_MODEL_DIR/digit_bert_base_joint_learning_ordered_split_num/digit_bert_base_joint_learning_tuning_batch_32_lr_5e-5_epoch_10_fold_4 \
    --save_strategy no \
    --overwrite_cache \
    --fp16 \
    --fp16_opt_level O2

time python3 $SRC_DIR/models/digit_bert_base_joint_learning/run_digit_joint_learning.py \
  --model_name_or_path $TUNED_MODEL_DIR/digit_bert_base_joint_learning_ordered_split_num/digit_bert_base_joint_learning_tuning_batch_32_lr_5e-5_epoch_10_fold_4 \
  --train_file $TRAIN_DIR/preprocess_digit_with_category/fold4/train.json \
  --validation_file $TRAIN_DIR/preprocess_digit_with_category/fold4/val.json \
  --test_file $TRAIN_DIR/preprocess_digit_with_category/fold4/val.json \
  --do_predict \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --output_dir $RESULT_DIR/digit_bert_base_joint_learning_ordered_split_num/digit_bert_base_joint_learning_tuning_batch_32_lr_5e-5_epoch_10_fold_4 \
  --overwrite_cache

python3 $SRC_DIR/evaluate/evaluate.py \
  --prediction_file $RESULT_DIR/digit_bert_base_joint_learning_ordered_split_num/digit_bert_base_joint_learning_tuning_batch_32_lr_5e-5_epoch_10_fold_4/predict_results_None.txt \
  --data_file $TRAIN_DIR/split_num/fold4/val.json \
  --result_file $RESULT_DIR/digit_bert_base_joint_learning_ordered_split_num/digit_bert_base_joint_learning_tuning_batch_32_lr_5e-5_epoch_10_fold_4/metrics.txt \
  --eval_validation \
  --merged_prediction

time python3 $SRC_DIR/models/digit_bert_base_joint_learning/run_digit_joint_learning.py \
  --model_name_or_path $TUNED_MODEL_DIR/digit_bert_base_joint_learning_ordered_split_num/digit_bert_base_joint_learning_tuning_batch_32_lr_5e-5_epoch_10_fold_4 \
  --train_file $TRAIN_DIR/preprocess_digit_with_category/fold4/train.json \
  --validation_file $TRAIN_DIR/preprocess_digit_with_category/fold4/val.json \
  --test_file $DEV_DIR/dev_preprocess_digit_with_category.json \
  --do_predict \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --output_dir $RESULT_DIR/digit_bert_base_joint_learning_ordered_split_num/digit_bert_base_joint_learning_tuning_batch_32_lr_5e-5_epoch_10_fold_4/dev \
  --overwrite_cache

python3 $SRC_DIR/evaluate/evaluate.py \
  --prediction_file $RESULT_DIR/digit_bert_base_joint_learning_ordered_split_num/digit_bert_base_joint_learning_tuning_batch_32_lr_5e-5_epoch_10_fold_4/dev/predict_results_None.txt \
  --data_file $DEV_DIR/dev_num.json \
  --result_file $RESULT_DIR/digit_bert_base_joint_learning_ordered_split_num/digit_bert_base_joint_learning_tuning_batch_32_lr_5e-5_epoch_10_fold_4/dev/metrics.txt \
  --merged_prediction \
  --unmerged_data_file $DEV_DIR/dev.json

time python3 $SRC_DIR/models/digit_bert_base_joint_learning/run_digit_joint_learning.py \
  --model_name_or_path $TUNED_MODEL_DIR/digit_bert_base_joint_learning_ordered_split_num/digit_bert_base_joint_learning_tuning_batch_32_lr_5e-5_epoch_10_fold_4 \
  --train_file $TRAIN_DIR/preprocess_digit_with_category/fold4/train.json \
  --validation_file $TRAIN_DIR/preprocess_digit_with_category/fold4/val.json \
  --test_file $TEST_DIR/test_preprocess_digit_with_category.json \
  --do_predict \
  --max_seq_length 512 \
  --per_device_eval_batch_size 16 \
  --output_dir $RESULT_DIR/digit_bert_base_joint_learning_ordered_split_num/digit_bert_base_joint_learning_tuning_batch_32_lr_5e-5_epoch_10_fold_4/test \
  --overwrite_cache

