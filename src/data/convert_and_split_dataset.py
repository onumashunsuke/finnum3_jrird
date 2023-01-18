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
"""
Preprocess and split training dataset.
Split dataset without shuffle.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import os
import random

from utils.utils import read_json_file, write_data_file_in_datasets, \
    extract_sentnece_and_labels
from preprocess_digit_to_digit import digitize_numerals
from preprocess_scientific_notation import convert_numerals_scientific_notation
from preprocess_markers import mask_numeral, put_numeral_between_markres


def _split_range(data: list, fold_k: int):
    """calculate data ranges of each fold
    """
    data_len = len(data)
    fold_len = [data_len // fold_k for _ in range(fold_k)]
    for i in range(data_len % fold_k):
        fold_len[i] += 1
    fold_range = []
    st = 0
    for fl in fold_len:
        fold_range.append((st, st + fl))
        st += fl
    return fold_range


def split_train_val(data: list, fold_range:list, val_fold: int):
    """split data records into val and training
    """
    (val_st, val_ed) = fold_range[val_fold]
    val = [data[i] for i in range(len(data)) if val_st <= i < val_ed]
    train = [data[i] for i in range(len(data)) if i < val_st or val_ed <= i]
    return train, val


def main(args: argparse.ArgumentParser):
    """split input data and write into k files in output directory

    Args:
        args (argparse.ArgumentParser):
        optional arguments:
            -h, --help            show this help message and exit
            --in_file IN_FILE     path for input data file
            --out_dir OUT_DIR     path for output data directory
            --split_k SPLIT_K     the number of split data sets
            --preprocess          whether to do preprocess on data and output for datasets
            --claim_only          whether to use claim label only, otherwise to use claim and category labels. if
                                    preprocess is false, this flag is ignored.
            --mask_numeral        whether to mask target numeral position for baseline models
            --put_numeral_between_markers
                                    whether to put target numeral between markers for baseline models
            --convert_scientific_notation
                                    whether to convert numeral into scientific notation
            --significant_figure_in_scientific_notation SIGNIFICANT_FIGURE_IN_SCIENTIFIC_NOTATION
                                    significant figure for scientific notation. e.g. when it is 4, 1234.5 => 1.234 [EXP] 3
            --split_digit         whether to digitize numeral
            --model_type {bert,roberta}
                                    mask and numeral marker token type for preprocess. bert = [MASK] for masking, [NUM] for
                                    numeral marker and [EXP] for scientific notation, roberta = <mask> for masking, <num>
                                    for numeral marker and <exp> for scientific notation
            --shuffle_train       whether to shuffle train dataset
            --random_seed RANDOM_SEED
                                    the ranodm seed for shuffle dataset
    """
    random.seed(args.random_seed)

    # print("split_dataset")
    # print("args:{}".format(args))

    # check argument for preprocess
    assert not (args.mask_numeral and args.split_digit), "Args error. mask numeral and split digit are exclusive."
    assert not (args.mask_numeral and args.convert_scientific_notation), "Args error. mask numeral and convert scientific notation are exclusive."
    assert not (args.mask_numeral and args.put_numeral_between_markers), "Args error. mask numeral and marking numeral are exclusive."
    data = read_json_file(args.in_file)

    mask_token = "[MASK]" if args.model_type == "bert" else "<mask>"
    marker_token = "[NUM]" if args.model_type == "bert" else "<num>"
    exp_token = "[EXP]" if args.model_type == "bert" else "<exp>"

    # make output directory
    split_dir = args.out_dir
    if os.path.exists(split_dir):
        assert False, "ERROR: output dir {} is exists. ".format(split_dir)
    os.mkdir(split_dir)
    for i in range(args.split_k):
        os.mkdir(os.path.join(split_dir, "fold{}".format(i)))

    # get fold range
    fold_range = _split_range(data, args.split_k)
    for i in range(args.split_k):
        fold_dir = os.path.join(split_dir, "fold{}".format(i))
        train_data, val_data = split_train_val(data, fold_range, i)

        # if mask_numeral => masking
        # else if only put numeral between markers => puting marker only
        # else if only digitize => digitize numeral and putting marker
        # else if convert scientific notation => converting numerals into scientific notation with marker

        if args.mask_numeral:
            train_data = mask_numeral(train_data, mask_token)
            val_data = mask_numeral(val_data, mask_token)
        elif args.put_numeral_between_markers and not (args.split_digit or args.convert_scientific_notation): # no digitize and scientific notation
            train_data = put_numeral_between_markres(train_data, marker_token=marker_token)
            val_data = put_numeral_between_markres(val_data, marker_token=marker_token)
        elif args.split_digit and not args.convert_scientific_notation:
            # only digitize
            # put numeral between markers by default
            train_data = digitize_numerals(train_data, marker_token=marker_token)
            val_data = digitize_numerals(val_data, marker_token=marker_token)
        elif args.convert_scientific_notation:
            # convert scientific notation
            sig = int(args.significant_figure_in_scientific_notation)
            assert sig > 0, "Args error. significant figure in scientific notation should be larger than 0"
            dig = args.split_digit
            train_data = convert_numerals_scientific_notation(train_data,
                marker_token=marker_token,
                significant=sig,
                digitize=dig,
                exp_marker=exp_token)
            val_data = convert_numerals_scientific_notation(val_data,
                marker_token=marker_token,
                significant=sig,
                digitize=dig,
                exp_marker=exp_token)
        if args.preprocess:
            train_data = extract_sentnece_and_labels(train_data, claim_only=args.claim_only)
            val_data = extract_sentnece_and_labels(val_data, claim_only=args.claim_only)
        if args.shuffle_train:
            random.shuffle(train_data)
        write_data_file_in_datasets(train_data, os.path.join(fold_dir, "train.json"))
        write_data_file_in_datasets(val_data, os.path.join(fold_dir, "val.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", help="path for input data file", required=True)
    parser.add_argument("--out_dir", help="path for output data directory", required=True)
    parser.add_argument("--split_k", help="the number of split data sets", default=5)
    parser.add_argument("--preprocess", help="whether to do preprocess on data and output for datasets", action="store_true")
    parser.add_argument("--claim_only", help="whether to use claim label only, otherwise to use claim and category labels. " + \
                                            "if preprocess is false, this flag is ignored.", action="store_true")
    parser.add_argument("--mask_numeral", help="whether to mask target numeral position for baseline models", action="store_true")
    parser.add_argument("--put_numeral_between_markers", help="whether to put target numeral between markers for baseline models", action="store_true")
    parser.add_argument("--convert_scientific_notation", help="whether to convert numeral into scientific notation", action="store_true")
    parser.add_argument("--significant_figure_in_scientific_notation", help="significant figure for scientific notation. e.g. when it is 4, 1234.5 => 1.234 [EXP] 3", default=4)
    parser.add_argument("--split_digit", help="whether to digitize numeral", action="store_true")
    parser.add_argument("--model_type", help="mask and numeral marker token type for preprocess. bert = [MASK] for masking, [NUM] for numeral marker and [EXP] for scientific notation, roberta = <mask> for masking, <num> for numeral marker and <exp> for scientific notation", choices=["bert", "roberta"], default="bert")
    parser.add_argument("--shuffle_train", help="whether to shuffle train dataset", action="store_true")
    parser.add_argument("--random_seed", help="the ranodm seed for shuffle dataset", default=42)
    args = parser.parse_args()
    main(args)
