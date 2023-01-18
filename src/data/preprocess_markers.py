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
"""preprocess for masking and marking
"""

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
        prefix_text = paragraph[:st]
        suffix_text = paragraph[ed:]
        new_paragraph = ""
        new_paragraph += prefix_text
        if len(prefix_text) > 0 and prefix_text[-1] != " ":
            # add space after prefix text
            new_paragraph += " "
        new_paragraph += mask_token
        if len(suffix_text) > 0 and suffix_text[0] != " ":
            # add space before suffix text
            new_paragraph += " "
        new_paragraph += suffix_text
        mask_d = d.copy()
        mask_d["paragraph"] = new_paragraph
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
        prefix_text = paragraph[:st]
        suffix_text = paragraph[ed:]
        new_paragraph = ""
        new_paragraph += prefix_text
        if len(prefix_text) > 0 and prefix_text[-1] != " ":
            # add space after prefix text
            new_paragraph += " "
        new_paragraph += marker_token + " " + paragraph[st:ed] + " " + marker_token
        if len(suffix_text) > 0 and suffix_text[0] != " ":
            # add space before suffix text
            new_paragraph += " "
        new_paragraph += suffix_text
        mark_d = d.copy()
        mark_d["paragraph"] = new_paragraph
        return mark_d
    marked = [_mark_paragraph(d) for d in data]
    return marked
