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

time python3 $SRC_DIR/models/scientific_t5_joint_learning/run_scientific_t5_joint_learning.py \
  --model_name_or_path           $BASE_MODEL_DIR/t5-large \
  --output_dir                   $TUNED_MODEL_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2 \
  --overwrite_output_dir         \
  --logging_dir                  $TUNED_MODEL_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/log \
  --save_strategy                no \
  \
  --max_source_length            512 \
  --max_target_length            10 \
  \
  --train_file                   $TRAIN_DIR/preprocess_roberta_scientific1_digit_with_category/fold2/train.json \
  --per_device_train_batch_size  8 \
  --gradient_accumulation_steps  4 \
  --learning_rate                5e-5 \
  --num_train_epochs             5 \
  --label_smoothing_factor       0.1 \
  --task_type joint_learning \
  \
  --test_file                    $TRAIN_DIR/preprocess_roberta_scientific1_digit_with_category/fold2/val.json \
  --pred_file                    $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/predict.txt \
  --num_beams                    5 \
  --per_device_eval_batch_size   16

python3 $SRC_DIR/evaluate/t5_predict_convert.py \
  --prediction_file $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/predict.txt \
  --result_file $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/predict_results_None.txt

python3 $SRC_DIR/evaluate/evaluate.py \
  --prediction_file $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/predict_results_None.txt \
  --data_file $TRAIN_DIR/split_num/fold2/val.json \
  --result_file $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/metrics.txt \
  --eval_validation \
  --merged_prediction

time python3 $SRC_DIR/models/scientific_t5_joint_learning/run_scientific_t5_joint_learning.py \
  --model_name_or_path           $TUNED_MODEL_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2 \
  --output_dir                   $TUNED_MODEL_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2 \
  \
  --max_source_length            512 \
  --max_target_length            10 \
  \
  --test_file                    $DEV_DIR/dev_preprocess_roberta_scientific1_digit_with_category.json \
  --pred_file                    $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/dev/predict.txt \
  --num_beams                    5 \
  --per_device_eval_batch_size   16 \
  --task_type joint_learning

python3 $SRC_DIR/evaluate/t5_predict_convert.py \
  --prediction_file $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/dev/predict.txt \
  --result_file $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/dev/predict_results_None.txt

python3 $SRC_DIR/evaluate/evaluate.py \
  --prediction_file $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/dev/predict_results_None.txt \
  --data_file $DEV_DIR/dev_num.json \
  --result_file $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/dev/metrics.txt \
  --merged_prediction \
  --unmerged_data_file $DEV_DIR/dev.json

time python3 $SRC_DIR/models/scientific_t5_joint_learning/run_scientific_t5_joint_learning.py \
  --model_name_or_path           $TUNED_MODEL_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2 \
  --output_dir                   $TUNED_MODEL_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2 \
  \
  --max_source_length            512 \
  --max_target_length            10 \
  \
  --test_file                    $TEST_DIR/test_preprocess_roberta_scientific1_digit_with_category.json \
  --pred_file                    $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/test/predict.txt \
  --num_beams                    5 \
  --per_device_eval_batch_size   16 \
  --task_type joint_learning

python3 $SRC_DIR/evaluate/t5_predict_convert.py \
  --prediction_file $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/test/predict.txt \
  --result_file $RESULT_DIR/scientific_t5_joint_learning_significant1_digit_ordered_split_num/scientific_t5_joint_learning_tuning_batch_32_lr_5e-5_epoch_5_fold_2/test/predict_results_None.txt

