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
"""PyTorch RoBERTa model. """
"""Adjust for FinNum-3 by The Japan Research Institute, Limited.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.roberta import RobertaPreTrainedModel, RobertaModel


class RobertaForSequenceClassificationJL(RobertaPreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config.keys_to_ignore_at_inference = ["loss1", "loss2"]
        self.num_labels1 = self.config.claim_num_labels
        self.num_labels2 = self.config.category_num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier1 = RobertaClassificationHead(config, num_labels=self.num_labels1)
        self.classifier2 = RobertaClassificationHead(config, num_labels=self.num_labels2)

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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
        sequence_output = outputs[0]
        logits1 = self.classifier1(sequence_output)
        logits2 = self.classifier2(sequence_output)

        self.config.problem_type = "single_label_classification"
        loss_fct = CrossEntropyLoss()
        loss1 = loss_fct(logits1.view(-1, self.num_labels1), label1.view(-1))
        loss2 = loss_fct(logits2.view(-1, self.num_labels2), label2.view(-1))

        return SequenceClassifierJLOutput(
            loss1=loss1,
            loss2=loss2,
            logits1=logits1,
            logits2=logits2,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels=None):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if num_labels is None:
            num_labels = config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



@dataclass
class SequenceClassifierJLOutput(SequenceClassifierOutput):
    loss1: Optional[torch.FloatTensor] = None
    loss2: Optional[torch.FloatTensor] = None
    logits1: torch.FloatTensor = None
    logits2: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None