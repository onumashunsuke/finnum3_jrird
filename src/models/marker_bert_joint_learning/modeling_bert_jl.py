# coding=utf-8
# Copyright 2022 The Japan Research Institute, Limited.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model. """
"""Adjust for FinNum-3 by The Japan Research Institute, Limited.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class BertForSequenceClassificationJL(BertPreTrainedModel):
    """FinNum3 BERT joint learning model
    joint leaning of claim detection and category classification
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config.keys_to_ignore_at_inference = ["loss1", "loss2"]
        self.num_labels1 = self.config.claim_num_labels
        self.num_labels2 = self.config.category_num_labels

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout1 = nn.Dropout(classifier_dropout)
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.classifier1 = nn.Linear(config.hidden_size, self.num_labels1)
        self.classifier2 = nn.Linear(config.hidden_size, self.num_labels2)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label1=None,
        label2=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        output1 = self.dropout1(pooled_output)
        output2 = self.dropout2(pooled_output)
        logits1 = self.classifier1(output1)
        logits2 = self.classifier2(output2)

        self.config.problem_type = "single_label_classification"
        loss_fct = CrossEntropyLoss()
        loss1 = loss_fct(logits1.view(-1, self.num_labels1), label1.view(-1))
        loss2 = loss_fct(logits2.view(-1, self.num_labels2), label2.view(-1))

        # if not return_dict:
        #     output = (logits1, logits2) + outputs[2:]
        #     return ((loss1, loss2) + output) if loss is not None else output

        return SequenceClassifierJLOutput(
            loss1=loss1,
            loss2=loss2,
            logits1=logits1,
            logits2=logits2,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class SequenceClassifierJLOutput(SequenceClassifierOutput):
    loss1: Optional[torch.FloatTensor] = None
    loss2: Optional[torch.FloatTensor] = None
    logits1: torch.FloatTensor = None
    logits2: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None