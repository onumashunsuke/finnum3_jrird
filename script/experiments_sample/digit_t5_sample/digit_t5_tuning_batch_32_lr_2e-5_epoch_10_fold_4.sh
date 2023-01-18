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

time python3 $SRC_DIR/models/digit_t5/run_digit_t5.py \
  --model_name_or_path           $BASE_MODEL_DIR/t5-large \
  --output_dir                   $TUNED_MODEL_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4 \
  --overwrite_output_dir         \
  --logging_dir                  $TUNED_MODEL_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/log \
  --save_strategy                no \
  \
  --max_source_length            512 \
  --max_target_length            10 \
  \
  --train_file                   $TRAIN_DIR/preprocess_roberta_digit/fold4/train.json \
  --per_device_train_batch_size  8 \
  --gradient_accumulation_steps  4 \
  --learning_rate                2e-5 \
  --num_train_epochs             10 \
  --label_smoothing_factor       0.1 \
  --task_type claim_only \
  \
  --test_file                    $TRAIN_DIR/preprocess_roberta_digit/fold4/val.json \
  --pred_file                    $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/predict.txt \
  --num_beams                    5 \
  --per_device_eval_batch_size   16

python3 $SRC_DIR/evaluate/t5_predict_convert.py \
  --prediction_file $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/predict.txt \
  --result_file $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/predict_results_None.txt \
  --claim_detect_only

python3 $SRC_DIR/evaluate/evaluate.py \
  --prediction_file $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/predict_results_None.txt \
  --data_file $TRAIN_DIR/split_num/fold4/val.json \
  --result_file $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/metrics.txt \
  --claim_detect_only \
  --eval_validation \
  --merged_prediction

time python3 $SRC_DIR/models/digit_t5/run_digit_t5.py \
  --model_name_or_path           $TUNED_MODEL_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4 \
  --output_dir                   $TUNED_MODEL_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4 \
  \
  --max_source_length            512 \
  --max_target_length            10 \
  \
  --test_file                    $DEV_DIR/dev_preprocess_roberta_digit.json \
  --pred_file                    $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/dev/predict.txt \
  --num_beams                    5 \
  --per_device_eval_batch_size   16 \
  --task_type claim_only

python3 $SRC_DIR/evaluate/t5_predict_convert.py \
  --prediction_file $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/dev/predict.txt \
  --result_file $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/dev/predict_results_None.txt \
  --claim_detect_only

python3 $SRC_DIR/evaluate/evaluate.py \
  --prediction_file $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/dev/predict_results_None.txt \
  --data_file $DEV_DIR/dev_num.json \
  --result_file $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/dev/metrics.txt \
  --claim_detect_only \
  --merged_prediction \
  --unmerged_data_file $DEV_DIR/dev.json

time python3 $SRC_DIR/models/digit_t5/run_digit_t5.py \
  --model_name_or_path           $TUNED_MODEL_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4 \
  --output_dir                   $TUNED_MODEL_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4 \
  \
  --max_source_length            512 \
  --max_target_length            10 \
  \
  --test_file                    $TEST_DIR/test_preprocess_roberta_digit.json \
  --pred_file                    $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/test/predict.txt \
  --num_beams                    5 \
  --per_device_eval_batch_size   16 \
  --task_type claim_only

python3 $SRC_DIR/evaluate/t5_predict_convert.py \
  --prediction_file $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/test/predict.txt \
  --result_file $RESULT_DIR/digit_t5_ordered_split_num/digit_t5_tuning_batch_32_lr_2e-5_epoch_10_fold_4/test/predict_results_None.txt \
  --claim_detect_only

