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
"""common preprocess for numerals
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from decimal import Decimal, ROUND_DOWN
import re

from utils.utils import read_json_file, write_data_file

SCALES = [
    ("hundred",     100),
    ("thousand",    1_000),
    ("million",     1_000_000),
    ("billion",     1_000_000_000),
    ("trillion",    1_000_000_000_000),
    ("quadrillion", 1_000_000_000_000_000),
]

def _is_wrong_point(text, num):
    """detect a period at the end of sentence
    """
    if num["target_num"][-1] != '.':
        return False
    pend = num["offset_end"]
    return len(text) == pend or text[pend] == ' ' or text[pend] == '\'' or text[pend] == '\"'


def _remove_suffixed_period(sentence_data):
    """remove suffixed period and return new annotation
    """
    text = sentence_data["paragraph"]
    for d in sentence_data["numerals"]:
        if not _is_wrong_point(text, d):
            continue
        d["target_num"] = d["target_num"][:-1]
        d["offset_end"] -= 1
    return sentence_data


def _is_hyphen_prefixed(d):
    return d["target_num"][0] == '-'


def _remove_prefixed_hyphen(sentence_data):
    """remove prefixed hyphen and return new annotation
    """
    for d in sentence_data["numerals"]:
        if not _is_hyphen_prefixed(d):
            continue
        d["target_num"] = d["target_num"][1:]
        d["offset_start"] += 1
    return sentence_data



def _get_scale(text, num):
    """return scale following the target num
    """
    text
    st = num["offset_start"]
    rest = text[st:].lower()
    for (s, m) in SCALES:
        res = re.match(r"\A({} ?{}s?)([ -.?;'\"])".format(num["target_num"], s), rest)
        if res:
            return (res, m, s)
    return None


def _scale(num_str, coef):
    """multiply the number with 10^coef
    """
    d = Decimal(num_str)
    scaled_d = d * coef
    rounded_d = Decimal(str(scaled_d)).quantize(Decimal("1."), rounding=ROUND_DOWN)
    assert scaled_d == rounded_d, "don't match! num * coef = {} {}".format(scaled_d, rounded_d)
    return str(rounded_d)


def _scale_numerals(sentence_data):
    """convert numerals with following scale
    """
    text = sentence_data["paragraph"]
    cum_shift = 0
    new_numerals = []
    for num in sentence_data["numerals"]:
        num["offset_start"] += cum_shift
        num["offset_end"] += cum_shift
        res = _get_scale(text, num)
        if res:
            match, coef, _ = res
            st = num["offset_start"]
            ed = num["offset_end"]
            assert num["target_num"] == text[st:ed]
            # calculate new number
            new_num = _scale(num["target_num"], coef)
            num["target_num"] = new_num
            # adjust positions
            text = text[:st] + new_num + text[st + match.end(1):]
            num["offset_end"] = st + len(new_num)
            cum_shift += len(new_num) - match.end(1)
        new_numerals.append(num)
    sentence_data["paragraph"] = text
    sentence_data["numerals"] = new_numerals
    return sentence_data


def _merge_neighboring_numerals(sentence_data):
    """merge neighboring numerals
    """
    new_numerals = [sentence_data["numerals"][0]]
    for num in sentence_data["numerals"][1:]:
        prev_num = new_numerals[-1]
        if num["offset_start"] == prev_num["offset_end"]:
            assert re.match(r".*[0-9]\Z", prev_num["target_num"]) is not None
            assert re.match(r"\A[0-9].*", num["target_num"]) is not None
            assert prev_num["claim"] == num["claim"]
            assert prev_num["category"] == num["category"]
            prev_num["target_num"] = prev_num["target_num"] + num["target_num"]
            prev_num["offset_end"] = num["offset_end"]
            prev_num["id"].extend(num["id"])
        else:
            new_numerals.append(num)
    sentence_data["numerals"] = new_numerals


def _expand_numerals(sentence_data):
    """expand annotaiton by including numbers before/after target numeral
    """
    text = sentence_data["paragraph"]
    # annotated position set
    annotated_indice = set()
    for num in sentence_data["numerals"]:
        for i in range(num["offset_start"], num["offset_end"]):
            annotated_indice.add(i)
    for num in sentence_data["numerals"]:
        st = num["offset_start"]
        ed = num["offset_end"]
        # include numbers before annotation if there are
        for i in reversed(range(0, st)):
            if i not in annotated_indice and '0' <= text[i] <= '9':
                num["target_num"] = text[i] + num["target_num"]
                num["offset_start"] -= 1
                annotated_indice.add(i)
            else:
                break
        # include numbers after annotation if there are
        for i in range(ed, len(text)):
            if i not in annotated_indice and '0' <= text[i] <= '9':
                num["target_num"] = num["target_num"] + text[i]
                num["offset_end"] += 1
                annotated_indice.add(i)
            else:
                break
    return sentence_data


def _merge_bothside_numerals_of_slash_or_colon(sentence_data):
    """merge both side numerals of slash or colon into a annotation.
    assume at least one numeral is annotated.
    """
    def _is_both_annotated(match, inner_nums):
        if len(inner_nums) != 2:
            return False
        l = inner_nums[0]
        r = inner_nums[1]
        return l["target_num"] + "/" + r["target_num"] == match[0] or l["target_num"] + ":" + r["target_num"] == match[0]

    def _is_formar_annotated(match, inner_nums):
        if len(inner_nums) != 1:
            return False
        l = inner_nums[0]
        return match[0].startswith(l["target_num"] + "/") or match[0].startswith(l["target_num"] + ":")

    def _is_latter_annotated(match, inner_nums):
        if len(inner_nums) != 1:
            return False
        r = inner_nums[0]
        return match[0].endswith("/" + r["target_num"]) or match[0].endswith(":" + r["target_num"])

    for match in re.finditer(r"[0-9]+[/:][0-9]+", sentence_data["paragraph"]):
        span_st = match.start()
        span_ed = match.end()
        inner_nums = []
        for num in sentence_data["numerals"]:
            st = num["offset_start"]
            ed = num["offset_end"]
            if span_st <= st < span_ed and span_st < ed <= span_ed:
                inner_nums.append(num)
            assert not (st < span_st and span_ed < ed), "match:{} num:{}".format(match, num)
            assert not (st < span_st and span_st < ed <= span_ed), "match:{} num:{}".format(match, num)
            assert not (span_st <= st < span_ed and span_ed < ed), "match:{} num:{}".format(match, num)

        if _is_both_annotated(match, inner_nums):
            # merge latter annotation to former annotation
            assert inner_nums[0]["claim"] == inner_nums[1]["claim"]
            assert inner_nums[0]["category"] == inner_nums[1]["category"]
            assert inner_nums[0]["offset_start"] < inner_nums[1]["offset_start"]
            inner_nums[0]["target_num"] = match[0]
            inner_nums[0]["offset_end"] = inner_nums[1]["offset_end"]
            inner_nums[0]["id"].extend(inner_nums[1]["id"])
            sentence_data["numerals"].remove(inner_nums[1])
        elif _is_formar_annotated(match, inner_nums):
            # expand former annotation with latter numeral
            inner_nums[0]["target_num"] = match[0]
            inner_nums[0]["offset_end"] = match.end()
        elif _is_latter_annotated(match, inner_nums):
            # expand latter annotation with former numeral
            inner_nums[0]["target_num"] = match[0]
            inner_nums[0]["offset_start"] = match.start()
    return sentence_data


def _check_sentence_continuity(data):
    """check that same sentences (with different annotations) appear successively
    """
    sentences = [data[0]["paragraph"]]
    for d in data[1:]:
        if d["paragraph"] == sentences[-1]:
            continue
        sentences.append(d["paragraph"])
    sentence_set = set(d["paragraph"] for d in data)
    return set(sentences) == sentence_set


def preprocess(args):
    """preprocess and output processed datasets.
    preprocessing
    - remove a period at the end of annotation
    - remove a leading hyphen at the start of annotation
    - expand annotation by mergin the target number with unannotated numbers before/after it
    - merge two successive annotations
    - convert scales
    - merge both side numerals of slash or colon
    """
    data = read_json_file(args.in_file)
    for (i, d) in enumerate(data):
        d["id"] = [i] # id is used for restoring merged annotations to original annotations
    assert _check_sentence_continuity(data)
    # grouping data by sentence key
    grouped_data = []
    curr = {"paragraph": data[0]["paragraph"], "numerals": []}
    for d in data:
        if curr["paragraph"] == d["paragraph"]:
            curr["numerals"].append({c: d[c] for c in d.keys() if c != "paragraph"})
        else:
            curr["numerals"].sort(key=lambda f: f["offset_start"])
            grouped_data.append(curr)
            curr = {"paragraph": d["paragraph"], "numerals": [{c: d[c] for c in d.keys() if c != "paragraph"}]}
    grouped_data.append(curr)
    # preprocess each grouped data
    for d in grouped_data:
        # remove a period at the end of annotation
        _remove_suffixed_period(d)
        # remove a leading hyphen at the start of annotation
        _remove_prefixed_hyphen(d)
        # expand annotation by mergin the target number with unannotated numbers before/after it
        _expand_numerals(d)
        # merge two successive annotations
        _merge_neighboring_numerals(d)
        # convert scales
        _scale_numerals(d)
        # merge both side numerals of slash or colon
        _merge_bothside_numerals_of_slash_or_colon(d)

    # reformat
    preprocessed_data = []
    for gd in grouped_data:
        for num in gd["numerals"]:
            rec = {
                "paragraph": gd["paragraph"],
            }
            rec.update(num)
            preprocessed_data.append(rec)
    preprocessed_data.sort(key=lambda x: x["id"][0])

    # output
    write_data_file(preprocessed_data, args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", help="path for input data file", required=True)
    parser.add_argument("--out_file", help="path for output data directory", required=True)
    args = parser.parse_args()
    preprocess(args)