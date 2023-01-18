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
"""preprocess for dataset without splitting
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import random

from utils.utils import read_json_file, write_data_file_in_datasets, \
    extract_sentnece_and_labels
from preprocess_digit_to_digit import digitize_numerals
from preprocess_scientific_notation import convert_numerals_scientific_notation
from preprocess_markers import mask_numeral, put_numeral_between_markres


def main(args: argparse.ArgumentParser):
    """preprocess dataset file

    Args:
        args (argparse.ArgumentParser):
        optional arguments:
            -h, --help            show this help message and exit
            --in_file IN_FILE     path for input data file
            --out_file OUT_FILE   path for output data file
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
            --shuffle             whether to shuffle dataset
            --random_seed RANDOM_SEED
                                    the ranodm seed for shuffle dataset
    """
    random.seed(args.random_seed)
    # check argument for preprocess
    assert not (args.mask_numeral and args.split_digit), "Args error. mask numeral and split digit are exclusive."
    assert not (args.mask_numeral and args.convert_scientific_notation), "Args error. mask numeral and convert scientific notation are exclusive."
    assert not (args.mask_numeral and args.put_numeral_between_markers), "Args error. mask numeral and marking numeral are exclusive."

    data = read_json_file(args.in_file)
    mask_token = "[MASK]" if args.model_type == "bert" else "<mask>"
    marker_token = "[NUM]" if args.model_type == "bert" else "<num>"
    exp_token = "[EXP]" if args.model_type == "bert" else "<exp>"

    # if mask_numeral => masking
    # else if only put numeral between markers => puting marker only
    # else if only digitize => digitize numeral and putting marker
    # else if convert scientific notation => converting numerals into scientific notation with marker

    if args.mask_numeral:
        data = mask_numeral(data, mask_token)
    elif args.put_numeral_between_markers and not (args.split_digit or args.convert_scientific_notation): # no digitize and scientific notation
        data = put_numeral_between_markres(data, marker_token=marker_token)
    elif args.split_digit and not args.convert_scientific_notation:
        # only digitize
        # put numeral between markers by default
        data = digitize_numerals(data, marker_token=marker_token)
    elif args.convert_scientific_notation:
        # convert scientific notation
        sig = int(args.significant_figure_in_scientific_notation)
        assert sig > 0, "Args error. significant figure in scientific notation should be larger than 0"
        dig = args.split_digit
        data = convert_numerals_scientific_notation(data,
            marker_token=marker_token,
            significant=sig,
            digitize=dig,
            exp_marker=exp_token)
    if args.preprocess:
        data = extract_sentnece_and_labels(data, args.claim_only)
    if args.shuffle:
        random.shuffle(data)
    write_data_file_in_datasets(data, args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", help="path for input data file", required=True)
    parser.add_argument("--out_file", help="path for output data file", required=True)
    parser.add_argument("--preprocess", help="whether to do preprocess on data and output for datasets", action="store_true")
    parser.add_argument("--claim_only", help="whether to use claim label only, otherwise to use claim and category labels. " + \
                                        "if preprocess is false, this flag is ignored.", action="store_true")
    parser.add_argument("--mask_numeral", help="whether to mask target numeral position for baseline models", action="store_true")
    parser.add_argument("--put_numeral_between_markers", help="whether to put target numeral between markers for baseline models", action="store_true")
    parser.add_argument("--convert_scientific_notation", help="whether to convert numeral into scientific notation", action="store_true")
    parser.add_argument("--significant_figure_in_scientific_notation", help="significant figure for scientific notation. e.g. when it is 4, 1234.5 => 1.234 [EXP] 3", default=4)
    parser.add_argument("--split_digit", help="whether to digitize numeral", action="store_true")
    parser.add_argument("--model_type", help="mask and numeral marker token type for preprocess. bert = [MASK] for masking, [NUM] for numeral marker and [EXP] for scientific notation, roberta = <mask> for masking, <num> for numeral marker and <exp> for scientific notation", choices=["bert", "roberta"], default="bert")
    parser.add_argument("--shuffle", help="whether to shuffle dataset", action="store_true")
    parser.add_argument("--random_seed", help="the ranodm seed for shuffle dataset", default=42)
    args = parser.parse_args()
    main(args)