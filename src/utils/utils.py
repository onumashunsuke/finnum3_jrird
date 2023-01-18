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

import json

TRAIN_RAW_FILE = "../../data/orig/FinNum-3_ConCall_train.json"
TRAIN_NUM_FILE = "../../data/num/FinNum-3_ConCall_train_num.json"
DEV_RAW_FILE = "../../data/orig/FinNum-3_ConCall_dev.json"
DEV_NUM_FILE = "../../data/num/FinNum-3_ConCall_dev_num.json"


def read_json_file(in_file: str) -> list:
    """read json file and return list of dictionaries
    """
    with open(in_file, "r") as f:
        data = json.load(f)
    return data


def write_data_file(data: list, out_file: str):
    """write data into out_file in json format
    """
    with open(out_file, "w") as f:
        json.dump(data, f)


def read_train_data():
    """read train data
    """
    return read_json_file(TRAIN_RAW_FILE)


def read_train_num_data():
    """read train data with digitize numerals
    """
    return read_json_file(TRAIN_NUM_FILE)


def read_dev_data():
    """read dev data
    """
    return read_json_file(DEV_RAW_FILE)


def read_dev_num_data():
    """read dev data with digitize numerals
    """
    return read_json_file(DEV_NUM_FILE)


def write_data_file_in_datasets(data: list, out_file: str, version:str = "1.0"):
    """write data into out_file in datasets json format
    """
    data = {
        "version": version,
        "data": data
    }
    with open(out_file, "w") as f:
        json.dump(data, f)


def get_categories_in_train_data() -> list:
    """get list of categories in train data
    """
    cat = [
        "price",
        "money",
        "change",
        "absolute",
        "relative",
        "date",
        "time",
        "quantity_absolute",
        "quantity_relative",
        "product number",
        "ranking",
        "other",
    ]
    return cat


def mask_numeral(data: list, mask_token="[MASK]") -> list:
    """mask target numeral with margin spaces

    Args:
        data (list): data read from json
        mask_token (str) : mask token

    Returns:
        list: processed data
    """
    def _mask_paragraph(d):
        paragraph = d["paragraph"]
        st = d["offset_start"]
        ed = d["offset_end"]
        mask_d = d.copy()
        mask_d["paragraph"] = paragraph[:st] + " " + mask_token + " " + paragraph[ed:]
        return mask_d
    masked = [_mask_paragraph(d) for d in data]
    return masked


def put_numeral_between_markres(data: list, marker_token="[NUM]") -> list:
    """pus target numeral between numeral markers

    Args:
        data (list): data read from json
        marker_token (str) : marker token

    Returns:
        list: processed data
    """
    def _mark_paragraph(d):
        paragraph = d["paragraph"]
        st = d["offset_start"]
        ed = d["offset_end"]
        mark_d = d.copy()
        mark_d["paragraph"] = paragraph[:st] + " " + marker_token + " " + paragraph[st:ed] + " " + marker_token + " " + paragraph[ed:]
        return mark_d
    marked = [_mark_paragraph(d) for d in data]
    return marked


def extract_sentnece_and_labels(data: list, claim_only: bool=True) -> list:
    """extract features
    when claim_only is true, extract only claim label.
    otherwise extract claim label and category label.

    Args:
        data (list): data read from json
        claim_only (bool): whether to use claim label only, otherwise to use claim and category labels.

    Returns:
        list: processed data
    """
    if claim_only:
        processed = [{"sentence1": d["paragraph"], "label": d["claim"]} for d in data]
    else:
        processed = [{"sentence1": d["paragraph"], "label1": d["claim"], "label2": d["category"]} for d in data]
    return processed