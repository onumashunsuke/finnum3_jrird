# README #

This repository is the resource of JRIRD team in NTCIR-16 FinNum-3 Manager's Claim Detection. For task details, please check the links below.

NTCIR-16 conference [NTCIR-16](https://research.nii.ac.jp/ntcir/ntcir-16/conference-ja.html)

FinNum-3 task [FinNum-3](https://sites.google.com/nlg.csie.ntu.edu.tw/finnum3/)

Our paper [Our paper](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings16/pdf/ntcir/05-NTCIR16-FINNUM-OnumaS.pdf)

## Directories
- script: Directory for scripts of data preprocessing, experiments, and evaluation
- src: Directory for sources

Directories below is needed for running scripts.

- data: Directory for data. There should be three subdirectories
    - orig: for original data
    - num: for intermidiate data
    - processed: for preprocessed data
- models: Directory for pre-trained and fine-tuned models
    - base: for pre-trained models
    - tuned: for fine-tuned models
- results: Directory for result files

## Running Examples
We show the procedure to run sample scripts below.

### Directory setup
- Make directories of data, models, and results.

~~~bash
mkdir -p data/orig
mkdir data/num
mkdir -p data/processed/ordered_split_num
mkdir data/processed/dev
mkdir data/processed/test
mkdir -p models/base
mkdir models/tuned
mkdir results
~~~

### Data Preparation
- Download the dataset from [NTCIR test collections page](https://research.nii.ac.jp/ntcir/data/data-en.html), and put three files under data/orig/
    - FinNum-3_ConCall_train.json
    - FinNum-3_ConCall_dev.json
    - FinNum-3_ConCall_test.json
- Run the preprocess script, preprocess_all.sh
    - Intermediate data are made in data/num
    - Preprocessed data are made in data/processed

~~~bash
cd script/data
bash preprocess_all.sh
~~~

Note that our preprocess includes a process to merge records.
The preprocessed datasets have different sizes from the original data.

### Model Preparation
- Download pre-trained models to use from [Huggingface](https://huggingface.co/models) and put these under models/base.
    - bert-base-cased
    - bert-large-cased
        - We used Whole Word Masking model
    - roberta-large
    - finbert
        - https://huggingface.co/ProsusAI/finbert
        - We remove the original classification layer from this model before fine-tuning it.
    - t5-large

### Train and Predict
- We extracted ten sample models from our experiments.
    - Please see the results in [our paper](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings16/pdf/ntcir/05-NTCIR16-FINNUM-OnumaS.pdf)
    - These scripts assume that only 1 GPU is visible.
    - Each model has five scripts for each training folds
        - For example, there are 5 script under script/experiments_sampe/digit_bert_base_sample.

| Task setting | PM | Format | Sample scripts |
| ---- | ---- | ---- | ---- |
| Claim Detection | BERT(base) | Digit | digit_bert_base_sample |
| Claim Detection | BERT(large) | Scientific(sig1) | scientific_bert_significant1_digit_sample |
| Claim Detection | FinBERT | Scientific(sig4) | scientific_finbert_significant4_digit_sample |
| Claim Detection | RoBERTa | Scientific(sig1) | scientific_roberta_significant1_digit_sample |
| Claim Detection | T5 | Digit | digit_t5_sample |
| Joint Learning | BERT(base) | Digit | digit_bert_base_joint_learning_sample |
| Joint Learning | BERT(large) | Marker | mask_roberta_joint_learning_sample |
| Joint Learning | FinBERT | Scientific(sig4) | scientific_finbert_joint_learning_significant4_digit_sample |
| Joint Learning | RoBERTa | Mask | mask_roberta_joint_learning_sample |
| Joint Learning | T5 | Scientific(sig1) | scientific_t5_joint_learning_significant1_digit_sample |


- Run these scripts for training and prediction.
    - digit_bert_base_tuning_batch_16_lr_2e-5_epoch_20_fold_0.sh
    - digit_bert_base_tuning_batch_32_lr_3e-5_epoch_15_fold_1.sh
    - digit_bert_base_tuning_batch_32_lr_2e-5_epoch_20_fold_2.sh
    - digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3.sh
    - digit_bert_base_tuning_batch_16_lr_3e-5_epoch_15_fold_4.sh
- As a result
    - Fine-tuned models are saved under models/tuned/digit_bert_base_ordered_split_num
    - Prediction files are saved under results/digit_bert_base_ordered_split_num


~~~bash
cd script/experiments_sample/digit_bert_base_sample
bash digit_bert_base_tuning_batch_16_lr_2e-5_epoch_20_fold_0.sh
bash digit_bert_base_tuning_batch_16_lr_2e-5_epoch_20_fold_0.sh
bash digit_bert_base_tuning_batch_32_lr_3e-5_epoch_15_fold_1.sh
bash digit_bert_base_tuning_batch_32_lr_2e-5_epoch_20_fold_2.sh
bash digit_bert_base_tuning_batch_32_lr_5e-5_epoch_20_fold_3.sh
bash digit_bert_base_tuning_batch_16_lr_3e-5_epoch_15_fold_4.sh
~~~

### Evaluation
- Our method averages predictions of five models as final prediction and output final prediction files.
- Run script/evaluate/evaluate_and_make_predictions.sh
    - The target model list to be evaluated should be in script/target_models_sasmple.json, but only digit_bert_base is in this sample.
    - We assume that the target models' results exist in the results directory.
    - for example, final prediction files are made under results/digit_bert_base_ordered_split_num
        - JRIRD_English_dev_x.json
        - JRIRD_English_dev_x_ddddd.jso
        - JRIRD_English_test_x.json

~~~bash
cd script/evaluate/
bash evaluate_and_make_prediction.sh
~~~

## Source description
Descriptions of each source file are in src/README.md

## Citation
~~~
Shunsuke Onuma and Kazuma Kadowaki. JRIRD at the NTCIR-16 FinNum-3 Task: Investigating the Effect of Numerical Representations in Manager’s Claim Detection. In Proceedings of the 16th NTCIR Conference on Evaluation of Information Access Technologies, pp. 108—115, June, 2022.
~~~