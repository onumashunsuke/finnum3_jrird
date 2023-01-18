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
"""convert numerals into digit-to-digit style
"""

def _digitize_numerals_for_record(rec, marker_token="[NUM]", all_digitize=True):
    """
    convert numerals into digit-to-digit style.

    Args:
        rec (dict): record of data
        marker_token (str, optional): marker of annotation. Defaults to "[NUM]".
        all_digitize (bool, optional): whether non target numerals are digitized. Defaults to True.
    """
    # xxx num xxx -> xxx [NUM] num [NUM] xxx
    text = rec["paragraph"]
    st = rec["offset_start"]
    ed = rec["offset_end"]
    temp_text = "$" + text[:st]
    if temp_text[-1] != " ": # there is not space before annotation, add it
        temp_text += " "
    temp_text += marker_token + " " + " ".join(list(rec["target_num"])) + " " + marker_token
    rest_text = text[ed:]
    if rest_text[0] != " ": # there is not space after annotation, add it
        temp_text += " "
    temp_text += rest_text

    if not all_digitize:
        return temp_text[1:]

    # convert non target numerals
    new_text = ['$']
    for ch in temp_text[1:]:
        if '0' <= ch <= '9' and new_text[-1] != ' ':
            new_text.append(' ') # if ch is number and there is not a space before it, add it
        elif ch != ' ' and '0' <= new_text[-1] <= '9':
            new_text.append(' ') # if pre ch is number and next ch is not number, add space
        new_text.append(ch)

    return "".join(new_text[1:])


def digitize_numerals(data, marker_token="[NUM]", all_digitize=True):
    new_data = []
    for rec in data:
        new_text = _digitize_numerals_for_record(rec, marker_token=marker_token, all_digitize=all_digitize)
        new_data.append({
            "paragraph": new_text,
            "claim": rec["claim"],
            "category": rec["category"],
            "id": rec["id"],
        })
    return new_data