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

SRC_DATA=../../src/data
INFILE=../../data/num/FinNum-3_ConCall_train_num.json
OUTDIR=../../data/processed/ordered_split_num

# only shuffle and split
python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/split_num \
    --shuffle_train


# mask numeral bert
python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_norm_mask \
    --mask_numeral \
    --claim_only \
    --preprocess \
    --shuffle_train \
    --model_type bert

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_norm_mask_with_category \
    --mask_numeral \
    --preprocess \
    --shuffle_train \
    --model_type bert

# mask numeral roberta
python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_norm_roberta_mask \
    --mask_numeral \
    --claim_only \
    --preprocess \
    --shuffle_train \
    --model_type roberta

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_norm_roberta_mask_with_category \
    --mask_numeral \
    --preprocess \
    --shuffle_train \
    --model_type roberta

# marker numeral bert
python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_marker \
    --put_numeral_between_markers \
    --claim_only \
    --preprocess \
    --shuffle_train \
    --model_type bert

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_marker_with_category \
    --put_numeral_between_markers \
    --preprocess \
    --shuffle_train \
    --model_type bert

# marker numeral roberta
python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_marker \
    --put_numeral_between_markers \
    --claim_only \
    --preprocess \
    --shuffle_train \
    --model_type roberta

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_marker_with_category \
    --put_numeral_between_markers \
    --preprocess \
    --shuffle_train \
    --model_type roberta


# digit numeral bert
python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_digit \
    --split_digit \
    --claim_only \
    --preprocess \
    --shuffle_train \
    --model_type bert

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_digit_with_category \
    --split_digit \
    --preprocess \
    --shuffle_train \
    --model_type bert

# digit numeral roberta
python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_digit \
    --split_digit \
    --claim_only \
    --preprocess \
    --shuffle_train \
    --model_type roberta

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_digit_with_category \
    --split_digit \
    --preprocess \
    --shuffle_train \
    --model_type roberta


# scientific notation significant1 bert
python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_scientific1 \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --claim_only \
    --preprocess \
    --shuffle_train \
    --model_type bert

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_scientific1_with_category \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --preprocess \
    --shuffle_train \
    --model_type bert

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_scientific1_digit \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --claim_only \
    --split_digit \
    --preprocess \
    --shuffle_train \
    --model_type bert

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_scientific1_digit_with_category \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --split_digit \
    --preprocess \
    --shuffle_train \
    --model_type bert

# scientific notation significant4 bert
python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_scientific4 \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --claim_only \
    --preprocess \
    --shuffle_train \
    --model_type bert

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_scientific4_with_category \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --preprocess \
    --shuffle_train \
    --model_type bert

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_scientific4_digit \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --claim_only \
    --split_digit \
    --preprocess \
    --shuffle_train \
    --model_type bert

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_scientific4_digit_with_category \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --split_digit \
    --preprocess \
    --shuffle_train \
    --model_type bert

# scientific notation significant1 roberta
python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_scientific1 \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --claim_only \
    --preprocess \
    --shuffle_train \
    --model_type roberta

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_scientific1_with_category \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --preprocess \
    --shuffle_train \
    --model_type roberta

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_scientific1_digit \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --claim_only \
    --split_digit \
    --preprocess \
    --shuffle_train \
    --model_type roberta

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_scientific1_digit_with_category \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --split_digit \
    --preprocess \
    --shuffle_train \
    --model_type roberta

# scientific notation significant4 roberta
python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_scientific4 \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --claim_only \
    --preprocess \
    --shuffle_train \
    --model_type roberta

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_scientific4_with_category \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --preprocess \
    --shuffle_train \
    --model_type roberta

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_scientific4_digit \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --claim_only \
    --split_digit \
    --preprocess \
    --shuffle_train \
    --model_type roberta

python ${SRC_DATA}/convert_and_split_dataset.py \
    --in_file  $INFILE \
    --out_dir $OUTDIR/preprocess_roberta_scientific4_digit_with_category \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --split_digit \
    --preprocess \
    --shuffle_train \
    --model_type roberta