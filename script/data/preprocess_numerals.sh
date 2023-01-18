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
DATA=../../data

INFILE_TRAIN=${DATA}/orig/FinNum-3_ConCall_train.json
OUTFILE_TRAIN=${DATA}/num/FinNum-3_ConCall_train_num.json

INFILE_DEV=${DATA}/orig/FinNum-3_ConCall_dev.json
OUTFILE_DEV=${DATA}/num/FinNum-3_ConCall_dev_num.json

INFILE_TEST=${DATA}/orig/FinNum-3_ConCall_test.json
OUTFILE_TEST=${DATA}/num/FinNum-3_ConCall_test_num.json

python ${SRC_DATA}/preprocess_numerals.py \
    --in_file  $INFILE_TRAIN \
    --out_file $OUTFILE_TRAIN

python ${SRC_DATA}/preprocess_numerals.py \
    --in_file  $INFILE_DEV \
    --out_file $OUTFILE_DEV

python ${SRC_DATA}/preprocess_numerals.py \
    --in_file  $INFILE_TEST \
    --out_file $OUTFILE_TEST