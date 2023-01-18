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
INFILE=../../data/num/FinNum-3_ConCall_test_num.json
ORIFILE=../../data/orig/FinNum-3_ConCall_test.json
OUTFILE=../../data/processed/test

# convert format only
python ${SRC_DATA}/convert_dataset.py \
    --in_file $ORIFILE \
    --out_file $OUTFILE/test.json

# convert num only
python ${SRC_DATA}/convert_dataset.py \
    --in_file $INFILE \
    --out_file $OUTFILE/test_num.json


# mask numeral bert
python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_norm_mask.json \
    --mask_numeral \
    --claim_only \
    --preprocess \
    --model_type bert

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_norm_mask_with_category.json \
    --mask_numeral \
    --preprocess \
    --model_type bert

# mask numeral roberta
python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_norm_roberta_mask.json \
    --mask_numeral \
    --claim_only \
    --preprocess \
    --model_type roberta

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_norm_roberta_mask_with_category.json \
    --mask_numeral \
    --preprocess \
    --model_type roberta

# marker numeral bert
python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_marker.json \
    --put_numeral_between_markers \
    --claim_only \
    --preprocess \
    --model_type bert

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_marker_with_category.json \
    --put_numeral_between_markers \
    --preprocess \
    --model_type bert

# marker numeral roberta
python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_marker.json \
    --put_numeral_between_markers \
    --claim_only \
    --preprocess \
    --model_type roberta

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_marker_with_category.json \
    --put_numeral_between_markers \
    --preprocess \
    --model_type roberta


# digit numeral bert
python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_digit.json \
    --split_digit \
    --claim_only \
    --preprocess \
    --model_type bert

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_digit_with_category.json \
    --split_digit \
    --preprocess \
    --model_type bert

# digit numeral roberta
python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_digit.json \
    --split_digit \
    --claim_only \
    --preprocess \
    --model_type roberta

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_digit_with_category.json \
    --split_digit \
    --preprocess \
    --model_type roberta


# scientific notation significant1 bert
python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_scientific1.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --claim_only \
    --model_type bert

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_scientific1_with_category.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --model_type bert

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_scientific1_digit.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --split_digit \
    --claim_only \
    --model_type bert

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_scientific1_digit_with_category.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --split_digit \
    --model_type bert



# scientific notation significant4 bert
python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_scientific4.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --claim_only \
    --model_type bert

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_scientific4_with_category.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --model_type bert

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_scientific4_digit.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --split_digit \
    --claim_only \
    --model_type bert

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_scientific4_digit_with_category.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --split_digit \
    --model_type bert


# scientific notation significant1 roberta
python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_scientific1.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --claim_only \
    --model_type roberta

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_scientific1_with_category.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --model_type roberta

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_scientific1_digit.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --split_digit \
    --claim_only \
    --model_type roberta

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_scientific1_digit_with_category.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 1 \
    --split_digit \
    --model_type roberta


# scientific notation significant4 roberta
python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_scientific4.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --claim_only \
    --model_type roberta

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_scientific4_with_category.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --model_type roberta

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_scientific4_digit.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --split_digit \
    --claim_only \
    --model_type roberta

python ${SRC_DATA}/convert_dataset.py \
    --in_file  $INFILE \
    --out_file $OUTFILE/test_preprocess_roberta_scientific4_digit_with_category.json \
    --preprocess \
    --convert_scientific_notation \
    --significant_figure_in_scientific_notation 4 \
    --split_digit \
    --model_type roberta