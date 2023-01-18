# coding=utf-8
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
"""Output prediction for test data
Unmerge preprocessed record and reformat for submitting.
"""
import argparse
import json
import shutil

from sklearn.metrics import f1_score

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluate import unmerge_prediction, read_predictions
from utils.utils import read_json_file

TEST_ROOT = "../../results"

TRAIN_ORIGIN = "../../data/orig/FinNum-3_ConCall_train.json"

TEST_MERGED = "../../data/processed/test/test_num.json"
TEST_UNMERGED = "../../data/processed/test/test.json"
TEST_ORIGIN = "../../data/orig/FinNum-3_ConCall_test.json"

DEV_MERGED = "../../data/processed/dev/dev_num.json"
DEV_UNMERGED = "../../data/processed/dev/dev.json"
DEV_ORIGIN = "../../data/orig/FinNum-3_ConCall_dev.json"


def _output_predictions(data_type, pred_file, output_file, claim_only=True):
    """output predictions from pred_file with original data
    """
    assert data_type == "dev" or data_type == "test"
    if data_type == "dev":
        merged_file = DEV_MERGED
        unmerged_file = DEV_UNMERGED
    elif data_type == "test":
        merged_file = TEST_MERGED
        unmerged_file = TEST_UNMERGED

    preds = read_predictions(pred_file)
    merged_data = read_json_file(merged_file)["data"]
    unmerged_preds = unmerge_prediction(preds, merged_data)

    unmerged_data = read_json_file(unmerged_file)["data"]
    for d, pr in zip(unmerged_data, unmerged_preds):
        d["prediction"] = int(pr[1])
        if not claim_only:
            d["category_prediction"] = pr[2]
    with open(output_file, "w") as f:
        json.dump(unmerged_data, f)


def _check_data_identity(original_file, pred_file, claim_only=True):
    """check data consistency between original data file and pred file
    """
    original_data = read_json_file(original_file)
    pred_data = read_json_file(pred_file)

    for data_ori, data_pre in zip(original_data, pred_data):
        for k in data_ori.keys():
            assert data_ori[k] == data_pre[k]
        assert "prediction" in data_pre.keys()
        if not claim_only:
            assert "category_prediction" in data_pre.keys()


def _check_category(pred_file, claim_only):
    """check there is not unvalid category in pred file
    """
    train_data = read_json_file(TRAIN_ORIGIN)
    train_claim_label = set(td["claim"] for td in train_data)
    train_category_label = set(td["category"] for td in train_data)

    pred_data = read_json_file(pred_file)
    for pd in pred_data:
        assert pd["prediction"] in train_claim_label
        if not claim_only:
            assert pd["category_prediction"] in train_category_label


def _attach_dev_score(dev_output_file, claim_only):
    """attach dev score to file name
    """
    dev_data = read_json_file(dev_output_file)
    # claim_detection
    y_true = [d["claim"] for d in dev_data]
    y_pred = [d["prediction"] for d in dev_data]
    claim_macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    print("claim:", claim_macro_f1)

    if not claim_only:
        y_true = [d["category"] for d in dev_data]
        y_pred = [d["category_prediction"] for d in dev_data]
        print("category:", f1_score(y_true=y_true, y_pred=y_pred, average="macro"))

    attach_file_name, _ = os.path.splitext(dev_output_file)
    attach_file_name += "_{:.4f}".format(claim_macro_f1) + ".json"
    shutil.copyfile(dev_output_file, attach_file_name)


def output_models_prediction(target_model, model_id):
    """output submit file for target model
    """
    print("model:", target_model)
    claim_only = "joint_learning" not in target_model
    dev_pred_file = os.path.join(TEST_ROOT, target_model, "predict_results_average_dev.txt")
    dev_output_file = os.path.join(TEST_ROOT, target_model, "JRIRD_English_dev_{}.json".format(model_id))
    _output_predictions("dev", dev_pred_file, dev_output_file, claim_only=claim_only)
    _check_data_identity(DEV_ORIGIN, dev_output_file, claim_only=claim_only)
    _check_category(dev_output_file, claim_only=claim_only)
    _attach_dev_score(dev_output_file, claim_only=claim_only)

    test_pred_file = os.path.join(TEST_ROOT, target_model, "predict_results_average_test.txt")
    test_output_file = os.path.join(TEST_ROOT, target_model, "JRIRD_English_test_{}.json".format(model_id))
    _output_predictions("test", test_pred_file, test_output_file, claim_only=claim_only)
    _check_data_identity(TEST_ORIGIN, test_output_file, claim_only=claim_only)
    _check_category(test_output_file, claim_only=claim_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_list", help="path for target model list", required=True)
    args = parser.parse_args()

    with open(args.model_list, "r") as fin:
        models = json.load(fin)

    for target_model in models:
        output_models_prediction(target_model, "x") # "x" is dummy id
