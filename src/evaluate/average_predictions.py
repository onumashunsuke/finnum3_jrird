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
"""make average predictions for models.
search best model on dev in each fold, and output these hyperparameters.

"""
import argparse
import collections
import json
import glob
import os

import numpy as np
from scipy.special import softmax

TEST_ROOT = "../../results"
SCRIPT = "../../script/evaluate"


# Priority of category; major category in train set has high priority
CATEGORY_PRIORITY = {
    "price": 1,
    "time": 2,
    "ranking": 3,
    "quantity_relative": 4,
    "product number": 5,
    "change": 6,
    "other": 7,
    "absolute": 8,
    "quantity_absolute": 9,
    "money": 10,
    "date": 11,
    "relative": 12,
}


def _read_file(file: str):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split("\t") for line in lines]
        return lines


def _split_logits(task, rec):
    if task == "claim_only":
        return rec[0], rec[1:]  # id, logits1
    else:
        return rec[0], rec[1:3], rec[3:]  # id, logits1, logits2


def read_predictions(model_dir, pred_dir):
    """read predictions, detecting task setting depend on columns
    """
    pred_file = os.path.join(model_dir, pred_dir, "predict_results_None.txt")
    predictions = _read_file(pred_file)
    task_type = "claim_only" if len(predictions[0]) == 2 else "joint_learning"
    predictions = predictions[1:]
    if task_type == "claim_only":
        predictions = {int(p[0]): {"claim": int(p[1])} for p in predictions}
    else:
        predictions = {int(p[0]): {"claim": int(p[1]), "category": p[2]} for p in predictions}
    return predictions


def read_logits(model_dir, pred_dir, task_type):
    """read logits, detecting task setting depend on columns
    """
    logits_file = os.path.join(model_dir, pred_dir, "predict_logits_None.txt")
    with open(logits_file, "r") as f:
        lines = f.readlines()

    lines = [line.strip().split("\t") for line in lines]
    if task_type == "claim_only":
        line_split = [_split_logits(task_type, line) for line in lines]
        _, label = line_split[0]  # header
        logits = {int(sp[0]): list(map(float, sp[1])) for sp in line_split[1:]}
        return label, logits
    else:
        line_split = [_split_logits(task_type, line) for line in lines]
        _, label1, label2 = line_split[0]  # header
        logits1 = {int(sp[0]): list(map(float, sp[1])) for sp in line_split[1:]}
        logits2 = {int(sp[0]): list(map(float, sp[2])) for sp in line_split[1:]}
        return label1, logits1, label2, logits2


def _average_logits(label_and_logits):
    """calculate average predictions from multiple logits
    label_and_logits = [
        (labels from model1, logits from model1),
        (labels from model1, logits from model2),
        ...
    ]
    """
    # check data consistency
    for (label, logits) in label_and_logits:
        assert label_and_logits[0][0] == label
        assert len(label_and_logits[0][1]) == len(logits)
    label = label_and_logits[0][0]
    data_len = len(label_and_logits[0][1])

    results = {}
    for i in range(data_len):
        # rearrange data
        # [model1_logits,
        #  model2_logits,
        #  ...] for data i
        logits_list = np.array([logits[i] for (_, logits) in label_and_logits])
        # softmax in each model logits
        prob_list = softmax(logits_list, axis=1)
        # average across models
        prob_avg = np.average(prob_list, axis=0)
        results[i] = {"prob": prob_avg, "pred": label[np.argmax(prob_avg)]}
    return results


def _average_predictions_for_claim_detection(predictions):
    """calculate average predictions on t5 for claim detection by majority voting
    """
    # check data consistency
    for pred in predictions:
        assert len(predictions[0]) == len(pred)

    results = {}
    data_len = len(predictions[0])
    for i in range(data_len):
        preds = [model_prediction[i]["claim"] for model_prediction in predictions]
        cnt = collections.Counter(preds)
        assert 0 < len(cnt) <= 2
        assert cnt[0] + cnt[1] == 5  # fold num
        if cnt[0] >= cnt[1]:
            results[i] = {"pred": 0}
        else:
            results[i] = {"pred": 1}
    return results


def _average_predictions_for_joint_learning(predictions):
    """calculate average predictions on t5 for joint learning by majority voting
    """
    # check data consistency
    for pred in predictions:
        assert len(predictions[0]) == len(pred)
    # for claim detection
    results_claim = {}
    data_len = len(predictions[0])
    for i in range(data_len):
        preds = [model_prediction[i]["claim"] for model_prediction in predictions]
        cnt = collections.Counter(preds)
        assert 0 < len(cnt) <= 2
        assert cnt[0] + cnt[1] == 5  # fold num
        if cnt[0] >= cnt[1]:
            results_claim[i] = {"pred": 0}
        else:
            results_claim[i] = {"pred": 1}

    # for joint learning
    results_category = {}
    data_len = len(predictions[0])
    for i in range(data_len):
        preds = [model_prediction[i]["category"] for model_prediction in predictions]
        cnt = collections.Counter(preds)
        assert sum(cnt.values()) == 5  # fold num
        category_cnt = [(k, cnt[k], CATEGORY_PRIORITY[k]) for k in cnt.keys()]
        category_cnt.sort(key=lambda x: (x[1], x[2]), reverse=True)
        results_category[i] = {"pred": category_cnt[0][0]}

        # check
        choice = category_cnt[0]
        for unchoice in category_cnt[1:]:
            assert choice[1] >= unchoice[1]
            if choice[1] == unchoice[1]:
                assert choice[2] > unchoice[2]

    return results_claim, results_category


def _average_predictions(models: list, pred_dir: str, task_type: str = "claim_only", hard_average: bool = False):
    """output average predictions for models from each fold

    Args:
        models (list): target fold models
        pred_dir (str) : targe model dir, dev or test
        output_file (str): prediction file
        task_type (str): task type, claim_only or joint_learning
        hard_average (bool): whether or not calculate average prediction by hard average, default is False (soft average)
    """
    if hard_average:
        if task_type == "claim_only":
            predictions = [read_predictions(model, pred_dir) for model in models]
            results = _average_predictions_for_claim_detection(predictions)
            return results, {}
        else:
            predictions = [read_predictions(model, pred_dir) for model in models]
            results_claim, results_category = _average_predictions_for_joint_learning(predictions)
            return results_claim, results_category
    elif task_type == "claim_only":
        label_and_logits = [read_logits(model, pred_dir, task_type) for model in models]
        resutls = _average_logits(label_and_logits)
        return resutls, {}
    else:
        label_and_logits = [read_logits(model, pred_dir, task_type) for model in models]
        label_and_logits_claim = [(ll[0], ll[1]) for ll in label_and_logits]
        label_and_logits_category = [(ll[2], ll[3]) for ll in label_and_logits]
        results_claim = _average_logits(label_and_logits=label_and_logits_claim)
        results_category = _average_logits(label_and_logits=label_and_logits_category)
        return results_claim, results_category


def _choice_fold_bests(model_dir, target_task="claim_detection", target_metric="macro_f1"):
    """search best fold model from model_dir
    """
    def _get_metric(target_dir):
        with open(os.path.join(target_dir, "metrics.txt"), "r") as f:
            res = json.load(f)
        assert target_task in res.keys()
        return res[target_task][target_metric]

    def _choice_in_fold(fold_n):
        candidates = glob.glob(os.path.join(model_dir, "*_fold_{}".format(fold_n)))
        candidates = [(cand, _get_metric(cand)) for cand in candidates]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    print(model_dir)
    fold_n = 5
    return [_choice_in_fold(k) for k in range(fold_n)]


def _output_claim_only(result, fo):
    fo.write("index\tprediction\n")
    for i in range(len(result)):
        fo.write("{}\t{}\n".format(i, result[i]["pred"]))


def _output_with_category(result_claim, result_category, fo):
    fo.write("index\tprediction1\tprediction2\n")
    assert len(result_claim) == len(result_category)
    for i in range(len(result_claim)):
        fo.write("{}\t{}\t{}\n".format(i, result_claim[i]["pred"], result_category[i]["pred"]))


def _get_choiced_fold_models(target_model):
    """get best fold models from target_model dir
    """
    model_dir = os.path.join(TEST_ROOT, target_model)
    fold_models = [fold_model[0] for fold_model in _choice_fold_bests(model_dir=model_dir)]
    return [fold_model.split("/")[-1] for fold_model in fold_models]


def load_all_choiced_fold_models():
    """read choiced_folds_for_test.json
    """
    choiced_folds_json = os.path.join(SCRIPT, "choiced_folds_for_test.json")
    with open(choiced_folds_json, "r") as fin:
        dict = json.load(fin)
    return dict


def _load_choiced_fold_models(target_model, choiced_fold_models):
    """load choiced fold models from dictionary
    """
    tm = target_model
    if target_model.endswith("_ordered_split_num"):
        tm = target_model.replace("_ordered_split_num", "")
    assert tm in choiced_fold_models
    return choiced_fold_models[tm]


def output_average_predictions(target_model, choiced_fold_models=None):
    """output average predictions for target_model on dev/test set
    """
    task_type = "claim_only" if "joint_learning" not in target_model else "joint_learning"
    hard_average = False if "t5" not in target_model else True
    print("OUTPUT:", target_model, task_type, "soft" if not hard_average else "hard")

    if choiced_fold_models is None:
        fold_models = _get_choiced_fold_models(target_model)
    else:
        fold_models = _load_choiced_fold_models(target_model, choiced_fold_models)
    # check directory
    for fold_model in fold_models:
        fold_test_dir = os.path.join(TEST_ROOT, target_model, fold_model, "test")
        assert os.path.exists(fold_test_dir)
    # dev
    dev_fold_models = [os.path.join(TEST_ROOT, target_model, fold_model) for fold_model in fold_models]
    result_claim, result_category = _average_predictions(dev_fold_models, "dev", task_type, hard_average=hard_average)
    with open(os.path.join(TEST_ROOT, target_model, "predict_results_average_dev.txt"), "w") as f:
        if task_type == "claim_only":
            _output_claim_only(result_claim, f)
        else:
            _output_with_category(result_claim, result_category, f)
    # test
    test_fold_models = [os.path.join(TEST_ROOT, target_model, fold_model) for fold_model in fold_models]
    result_claim, result_category = _average_predictions(test_fold_models, "test", task_type, hard_average=hard_average)
    with open(os.path.join(TEST_ROOT, target_model, "predict_results_average_test.txt"), "w") as f:
        if task_type == "claim_only":
            _output_claim_only(result_claim, f)
        else:
            _output_with_category(result_claim, result_category, f)


def output_test_folds_for_models(models):
    """output choiced_folds data
    """
    result = {}
    for target_model in models:
        fold_models = _get_choiced_fold_models(target_model)
        target_name = target_model.replace("_ordered_split_num", "")
        result[target_name] = fold_models
    choiced_folds_json = os.path.join(SCRIPT, "choiced_folds_for_test.json")
    with open(choiced_folds_json, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_list", help="path for target model list", required=True)
    args = parser.parse_args()

    with open(args.model_list, "r") as fin:
        models = json.load(fin)

    output_test_folds_for_models(models)
    choiced_fold_models = load_all_choiced_fold_models()
    for model in models:
        output_average_predictions(model, choiced_fold_models=choiced_fold_models)
