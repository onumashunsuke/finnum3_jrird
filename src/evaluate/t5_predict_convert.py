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
"""convert t5 predictions
"""
import argparse

CLAIM_MAP = {
    "out claim": 0,
    "in claim": 1,
}

CATEGORY_MAP = {
    "price": "price",
    "money": "money",
    "change": "change",
    "absolute": "absolute",
    "relative": "relative",
    "date": "date",
    "time": "time",
    "quantity absolute": "quantity_absolute",
    "quantity relative": "quantity_relative",
    "product number": "product number",
    "ranking": "ranking",
    "other": "other",
}


def _convert_label_in_claim(s):
    if s not in CLAIM_MAP.keys():
        return 0
    return CLAIM_MAP[s]


def _convert_label_in_category(s):
    if s not in CATEGORY_MAP.keys():
        return "relative"  # majority category
    return CATEGORY_MAP[s]


def _convert_claim_only(pred_list):
    return [(i, _convert_label_in_claim(ps)) for i, ps in enumerate(pred_list)]


def _convert_joint_learning(pred_list):
    it = iter(pred_list)
    return [(i, _convert_label_in_claim(ps1), _convert_label_in_category(ps2)) for i, (ps1, ps2) in enumerate(zip(it, it))]


def _read_file(pred_file):
    with open(pred_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines


def main(args):
    """convert t5 predictions
    if there are invalid predictions, convert these into major class
    """
    pred_list = _read_file(args.prediction_file)
    if args.claim_detect_only:
        conv_list = _convert_claim_only(pred_list)
    else:
        conv_list = _convert_joint_learning(pred_list)

    with open(args.result_file, "w") as f:
        header = "index\tprediction\n" if args.claim_detect_only else "index\tprediction1\tprediction2\n"
        f.write(header)
        for line in conv_list:
            f.write("\t".join(map(str, line)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", help="path for prediction file", required=True)
    parser.add_argument("--result_file", help="path for convert output file", required=True)
    parser.add_argument("--claim_detect_only", help="whether prediction is only claim detection", action="store_true")
    args = parser.parse_args()
    main(args)