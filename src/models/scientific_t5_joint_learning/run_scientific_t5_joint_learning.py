#!/usr/bin/env python3
# coding=utf-8
# Copyright 2022 The Japan Research Institute, Limited.
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
"""Adjust for FinNum-3 by The Japan Research Institute, Limited.
"""
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List
import datasets
from datasets import set_caching_enabled
# import numpy as np
# from datasets import load_dataset
import transformers
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, HfArgumentParser, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
from transformers.trainer_utils import HubStrategy, IntervalStrategy
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# unuse datasets cache
set_caching_enabled(False)

check_min_version("4.10.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

@dataclass
class MySeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    do_train: bool = field(default=False, init=False)
    do_predict: bool = field(default=False, init=False)
    do_eval: bool = field(default=False, init=False)
    evaluation_strategy: IntervalStrategy = field(default="no", init=False)
    prediction_loss_only: bool = field(default=False, init=False)
    eval_accumulation_steps: Optional[int] = field(default=None, init=False)
    log_level: Optional[str] = field(default="info", init=False)
    log_level_replica: Optional[str] = field(default="warning", init=False)
    label_names: Optional[List[str]] = field(default=None, init=False)
    group_by_length: bool = field(default=False, init=False)
    length_column_name: Optional[str] = field(default="length", init=False)
    use_legacy_prediction_loop: bool = field(default=False, init=False)
    push_to_hub: bool = field(default=False, init=False)
    resume_from_checkpoint: Optional[str] = field(default=None, init=False)
    hub_model_id: str = field(default=None, init=False)
    hub_strategy: HubStrategy = field(default="every_save", init=False)
    hub_token: str = field(default=None, init=False)
    push_to_hub_model_id: str = field(default=None, init=False)
    push_to_hub_organization: str = field(default=None, init=False)
    push_to_hub_token: str = field(default=None, init=False)
    predict_with_generate: bool = field(default=True, init=False)
    generation_max_length: Optional[int] = field(default=None, init=False)
    generation_num_beams: Optional[int] = field(default=None, init=False)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(metadata={
        "help": "Path to pretrained model or model identifier from huggingface.co/models"
    })
    config_name: Optional[str] = field(default=None, metadata={
        "help": "Pretrained config name or path if not the same as model_name"
    })
    tokenizer_name: Optional[str] = field(default=None, metadata={
        "help": "Pretrained tokenizer name or path if not the same as model_name"
    })
    use_fast_tokenizer: bool = field(default=True, metadata={
        "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
    })

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(default=None, metadata={
        "help": "The input training data file (a jsonlines or csv file)."
    })
    test_file: Optional[str] = field(default=None, metadata={
        "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
    })
    pred_file: Optional[str] = field(default=None, metadata={
    })
    max_source_length: Optional[int] = field(default=1024, metadata={
        "help": "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded."
    })
    max_target_length: Optional[int] = field(default=128, metadata={
        "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded."
    })
    pad_to_max_length: bool = field(default=False, metadata={
        "help": "Whether to pad all samples to model maximum sentence length. "
        "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
        "efficient on GPU but very bad for TPU."
    })
    num_beams: Optional[int] = field(default=None, metadata={
        "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
        "which is used during ``evaluate`` and ``predict``."
    })
    task_type: Optional[str] = field(default="claim_only")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MySeq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    assert data_args.train_file is not None or data_args.test_file is not None, "There is nothing to do. Please pass `train_file` and/or `test_file`."
    training_args.do_train = data_args.train_file is not None
    training_args.do_predict = data_args.test_file is not None

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    assert training_args.overwrite_output_dir or not training_args.do_train or os.path.exists(training_args.output_dir) or not os.listdir(training_args.output_dir), f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    # Set seed before initializing model.

    set_seed(training_args.seed)
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer)
    # tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]"]})
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config)
    # model.resize_token_embeddings(len(tokenizer))
    assert model.config.decoder_start_token_id is not None, "Make sure that `config.decoder_start_token_id` is correctly defined"
    # Preprocessing the datasets.
    assert training_args.label_smoothing_factor <= 0 or hasattr(model, "prepare_decoder_input_ids_from_labels"), "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined."

    # add special tokens for task data (roberta style)
    num_added_toks = tokenizer.add_tokens(["<num>", "<exp>"], special_tokens=True)
    print('[INFO] We have added', num_added_toks, 'tokens')
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    model.resize_token_embeddings(len(tokenizer))

    def preprocess_function(examples):
        inputs = examples["src"]
        targets = examples["tgt"]
        # # prefix is already added.
        # inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding="max_length" if data_args.pad_to_max_length else False, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=data_args.max_target_length, padding="max_length" if data_args.pad_to_max_length else False, truncation=True)
        # ignore padding in the loss.
        if data_args.pad_to_max_length:
            labels["input_ids"] = [
                [(lbl if lbl != tokenizer.pad_token_id else -100) for lbl in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _read_dataset_claim_only(data_list):
            assert "label1" not in data_list[0].keys()
            prefix = "claim classification :"
            src, tgt = [], []
            for rec in data_list:
                src.append(prefix + rec["sentence1"])
                label_text = str(rec["label"])
                if label_text == "0":
                    label_text = "out claim"
                elif label_text == "1":
                    label_text = "in claim"
                tgt.append(label_text)
            return src, tgt


    def _read_dataset_joint_learning(data_list):
            assert "label1" in data_list[0].keys()
            prefix_claim = "claim classification :"
            prefix_category = "category classification :"
            src, tgt = [], []
            for rec in data_list:
                # for claim detection
                src.append(prefix_claim + rec["sentence1"])
                claim_text = str(rec["label1"])
                if claim_text == "0":
                    claim_text = "out claim"
                elif claim_text == "1":
                    claim_text = "in claim"
                tgt.append(claim_text)
                # for category detection
                src.append(prefix_category + rec["sentence1"])
                category_text = str(rec["label2"])
                category_text = category_text.replace("_", " ")
                tgt.append(category_text)
            return src, tgt

    def read_dataset(data_file):
        with open(data_file, "r", encoding="utf-8-sig") as fp:
            data = json.load(fp)["data"]
            if data_args.task_type == "claim_only":
                return _read_dataset_claim_only(data)
            else:
                return _read_dataset_joint_learning(data)

    # https://huggingface.co/docs/datasets/loading_datasets.html.
    if training_args.do_train:
        src, tgt = read_dataset(data_args.train_file)
        train_dataset = datasets.Dataset.from_dict({"src": src, "tgt": tgt})
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=None, remove_columns=train_dataset.column_names, load_from_cache_file=False, desc="Running tokenizer on train dataset")

    if training_args.do_predict:
        src, tgt = read_dataset(data_args.test_file)
        eval_dataset = datasets.Dataset.from_dict({"src": src, "tgt": tgt})
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=None, remove_columns=eval_dataset.column_names, load_from_cache_file=False, desc="Running tokenizer on prediction dataset")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8 if training_args.fp16 else None)
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_dataset if training_args.do_train else None, eval_dataset=eval_dataset if training_args.do_eval else None, tokenizer=tokenizer, data_collator=data_collator)
    # Training
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=None)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()
    # Evaluation
    if training_args.do_predict:
        logger.info("*** Predict ***")
        # make dir
        os.makedirs(os.path.dirname(data_args.pred_file), exist_ok=True)
        predict_results = trainer.predict(eval_dataset, metric_key_prefix="predict", max_length=data_args.max_target_length, num_beams=data_args.num_beams)
        if trainer.is_world_process_zero() and data_args.pred_file is not None:
            predictions = tokenizer.batch_decode(predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions = [pred.strip() for pred in predictions]
            with open(data_args.pred_file, "w") as writer:
                writer.write("\n".join(predictions))


if __name__ == "__main__":
    main()