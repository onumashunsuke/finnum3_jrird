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
"""evaluation script for finnum3
"""
import argparse
import json

from sklearn.metrics import f1_score


def read_predictions(pred_file: str):
    with open(pred_file, "r") as f:
        preds = f.readlines()
        preds = [line.strip().split("\t") for line in preds[1:]]  # skip header
    return preds


def _read_json_file(in_file: str) -> list:
    """read json file and return list of dictionaries
    """
    with open(in_file, "r") as f:
        data = json.load(f)
    return data


def _eval_claim_detection(preds: list, data: list) -> dict:
    """evaluation for claim detection
    assume pred[0] is id, and pred[1] is binary prediction

    Args:
        preds (list): prediction data, column 0 is id, column 1 is binary prediction
        data (list): original data that have claim label and category type

    Returns:
        dict: dictionary for evaluation metrics
    """
    y_pred = [int(p[1]) for p in preds]
    y_true = [int(d["claim"]) for d in data]
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    return {"macro_f1": macro_f1, "micro_f1": micro_f1}


def _eval_category_classification(preds: list, data: list) -> dict:
    """evaluation for category classification
    assume pred[0] is id, and pred[2] is label prediction

    Args:
        preds (list): prediction data, column 0 is id, column 2 is label prediction
        data (list): original data that have claim label and category type

    Returns:
        dict: dictionary for evaluation metrics
    """
    y_pred = [p[2] for p in preds]
    y_true = [d["category"] for d in data]
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    return {"macro_f1": macro_f1, "micro_f1": micro_f1}


def unmerge_prediction(preds: list, merged_data: list):
    """unmerge predictions based on merged ids in merged data file.
    """
    preds_with_ids = []
    assert len(preds) == len(merged_data)
    # unmerge preds with id
    for (pr, md) in zip(preds, merged_data):
        for i in md["id"]:
            preds_with_ids.append((i, pr))
    # sort by id and return predictions only
    preds_with_ids.sort(key=lambda x: x[0])
    return [p for (i, p) in preds_with_ids]


def _unmerge_data(data: list, merged_data: list):
    """unmerge data based on merged ids in merged data file.
    """
    data_with_ids = []
    assert len(data) == len(merged_data), "data:{}, merged_data:{}".format(len(data), len(merged_data))
    # unmerge preds with id
    for (d, md) in zip(data, merged_data):
        for i in md["id"]:
            data_with_ids.append((i, d))
    # sort by id and return predictions only
    data_with_ids.sort(key=lambda x: x[0])
    return [d for (i, d) in data_with_ids]


def main(args: argparse.ArgumentParser):
    preds = read_predictions(args.prediction_file)
    data = _read_json_file(args.data_file)["data"]
    if args.merged_prediction and args.unmerged_data_file == "":
        # data is also merged as predictions
        preds = unmerge_prediction(preds, data)
        data = _unmerge_data(data, data)
    if args.merged_prediction and args.unmerged_data_file != "":
        # because data is merged, predictions should be unmerged
        # and then compare with unmerged data
        preds = unmerge_prediction(preds, data)
        unmerged_data = _read_json_file(args.unmerged_data_file)["data"]
        data = unmerged_data
    assert len(preds) == len(data), "data size unmatched. prediction file or data file is wrong."
    if args.claim_detect_only:
        res = {"claim_detection": _eval_claim_detection(preds, data)}
    else:
        res = {
            "claim_detection": _eval_claim_detection(preds, data),
            "category_classification": _eval_category_classification(preds, data)
        }
    with open(args.result_file, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", help="path for input data file", required=True)
    parser.add_argument("--prediction_file", help="path for prediction file", required=True)
    parser.add_argument("--merged_prediction", help="whether predictions is merged", action="store_true")
    parser.add_argument("--unmerged_data_file", help="path for unmerged data file", default="")
    parser.add_argument("--eval_validation", help="whether to evaluate predictions for eval data", action="store_true")
    parser.add_argument("--result_file", help="path for result output file", required=True)
    parser.add_argument("--claim_detect_only", help="whether prediction is only claim detection", action="store_true")
    args = parser.parse_args()
    main(args)