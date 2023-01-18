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
"""convert numerals into scientific notation
ex) 123 -> 1.23e2
"""
from decimal import Decimal, ROUND_HALF_UP
import re


def _convert(num_str, significant, digitize, exp_marker="[EXP]"):
    """convert numeral into scientific notation using decimal type.
    if digitize = True, digitize exponential and mantissa parts
    """
    assert significant == 1 or significant == 4
    if significant == 1:
        sci_text = "{:.0e}".format(Decimal(num_str))
    elif significant == 4:
        sci_text = "{:.3e}".format(Decimal(num_str))
    sci_parts = sci_text.split("e")
    mantissa = sci_parts[0]
    exp = sci_parts[1]
    # remove +
    if exp[0] == '+':
        exp = exp[1:]
    if digitize:
        mantissa = " ".join(list(str(mantissa)))
        exp = " ".join(list(str(exp)))
    return " {} {} {} ".format(mantissa, exp_marker, exp)



def _convert_numeral(num_str, significant=None, digitize=True, exp_marker="[EXP]"):
    """convert numeral into scientific notation.
    scientific notation is a format like D.DDDeX, meaning d.ddd * 10 ^ x.
    significant digits should be 1 or 4.
    if digitize = True, digitize exponential and mantissa parts.

    if "/" or ":" appears in numeral, convert both side of it
        e.g.) "1/3" -> "1.0 e 1 / 3.0 e 1"

    Args:
        num_str (str): target numeral string (assume that it is treated in Decimal type)
        significant (int, optional): significant digits. it should be 1 or 4. Defaults to None.
        digitize (bool, optional): whether digitize the numbers in exponential and mantissa parts. Defaults to True.
        exp_marker (str, optional): sep marker for exp and mantisssa. Defaults to "[EXP]".

    Returns:
        str: converted string
    """
    sep = None
    if '/' in num_str:
        sep = '/'
    if ':' in num_str:
        sep = ':'

    if sep is not None:
        temp = num_str.split(sep)
        former = temp[0]
        latter = temp[1]
        return _convert(former, significant, digitize, exp_marker) + sep + _convert(latter, significant, digitize, exp_marker)
    else:
        return _convert(num_str, significant, digitize, exp_marker)


def _convert_all_numerals_in_text(text, significant=None, digitize=True, exp_marker="[EXP]"):
    """convert all numerals into scientific notation
    """
    pattern = r"([0-9]+\.[0-9]+|[0-9]+)"
    nums = re.findall(pattern, text)
    rest_text = text
    sub_strings = []
    for num in nums:
        match_obj = re.search(r"{}".format(num), rest_text)
        former = rest_text[:match_obj.start()]
        convert_num = _convert_numeral(num, significant, digitize, exp_marker)
        if len(former) > 0:
            sub_strings.append(former)
        sub_strings.append(convert_num)
        rest_text = rest_text[match_obj.end():]
    if rest_text != "":
        sub_strings.append(rest_text)
    result_text = "$"
    for sub in sub_strings:
        if sub[0] == " " and result_text[-1] == " ":
            result_text = result_text.rstrip()
        result_text += sub
    return result_text[1:]


def _convert_record(rec, marker_token="[NUM]", all_convert=True, significant=4, digitize=True, exp_marker="[EXP]"):
    """convert numerals into scientific notation.
    surround the target numeral with marker_token.
    if all_convert = True, convert other numerals into scientific_notation.

    Args:
        rec (list): a record of data
        marker_token (str, optional): marker for the target numeral. Defaults to "[NUM]".
        all_convert (bool, optional): whether or not convert all numerals. Defaults to True.
        significant (int, optional): significant digits. Defaults to 4.
        digitize (bool, optional): whether digitize the numbers in exponential and mantissa parts. Defaults to True.
        exp_marker (str, optional): marker of separation between exponential and mantissa parts. Defaults to "[EXP]".
    """
    # xxx num xxx -> xxx [NUM] num [NUM] xxx
    text = rec["paragraph"]
    st = rec["offset_start"]
    ed = rec["offset_end"]
    temp_text = "$" + text[:st]
    if temp_text[-1] != " ": # if there is not space before the target num, add it
        temp_text += " "
    temp_text += marker_token + _convert_numeral(rec["target_num"], significant=significant, digitize=digitize, exp_marker=exp_marker) + marker_token
    rest_text = text[ed:]
    if rest_text[0] != " ": # if there is not space after the target num, add it
        temp_text += " "
    temp_text += rest_text
    temp_text = temp_text[1:]

    if not all_convert:
        return temp_text

    # convert two text parts; text before the target numeral, and text after the target numeral
    splitted = temp_text.split(marker_token)
    former_text = splitted[0]
    latter_text = splitted[2]

    converted_text = _convert_all_numerals_in_text(former_text, significant, digitize, exp_marker)
    converted_text += marker_token + splitted[1] + marker_token
    converted_text += _convert_all_numerals_in_text(latter_text, significant, digitize, exp_marker)
    return converted_text


def convert_numerals_scientific_notation(data, marker_token="[NUM]", all_convert=True, significant=4, digitize=True, exp_marker="[EXP]"):
    new_data = []
    for rec in data:
        new_text = _convert_record(rec, marker_token=marker_token, all_convert=all_convert, significant=significant, digitize=digitize, exp_marker=exp_marker)
        new_data.append({
            "paragraph": new_text,
            "claim": rec["claim"],
            "category": rec["category"],
            "id": rec["id"],
        })
    return new_data


if __name__ == "__main__":
    pass