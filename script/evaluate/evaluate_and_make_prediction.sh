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

SRC_DATA=../../src/evaluate

# make average predictions
python3 $SRC_DATA/average_predictions.py \
    --model_list $(pwd)/target_models_sample.json

# make final prediction files
python3 $SRC_DATA/output_test_predictions.py \
    --model_list $(pwd)/target_models_sample.json