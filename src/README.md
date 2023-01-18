# Source Descriptions

Sources are put below.

- data: sources for data preprocessing
- models: sources for the experiment models
- evaluate: sources for the evaluation process
- utils: utility sources

## Data preprocess

There are three top-level scripts.
- preprocess_numerals.py
    - This script is for common preprocess for datasets
- convert_and_split_dataset.py
    - This script preprocesses the train set, splits data into folds, and converts the numbers into each format.
- convert_dataset.py
    - This script preprocesses the dev/test sets, and converts the numbers into each format.


The conversion processes for each numerical format are below.

- preprocess_markers.py
    - Process for mask and marker formats.
- preprocess_digit_to_digit.py
    - Process for digit format.
- preprocess_scientific_notation.py
    - Process for scientific format.

## Experiment Models

Sources of experiment models are put under models dir.

Directory names indicate a base language model, a numerical format, and a task setting, either Claim Detection Only or Joint Learning.
The naming rule is as follows.

- <format_name>_<base_model_name>
    - Models in Claim Detection Only setting
- <format_name>_<base_model_name>_joint_learning
    - Models in Joint Learning setting

Note that we created other directories and sources when the preprocess or the model size was only different. For example, we put the same scripts separately for BERT_large and BERT_base with different names.

### Claim Detection Only models (except for T5)
We created the scripts based on a classification example from Huggingface.
We added special tokens for each numerical format.

### Joint Learning models (except for T5)
We modified sources based on Claim Detection Only models.
Major changes for joint learning are as follows.

- Modeling (e.g., modeling_bert_jl.py)
    - Model receives two labels
    - Model has two classification layers
- Trainer (e.g., trainer_jl.py)
    - Trainer receives two losses and averages them as a joint learning loss

### T5
Since T5 is a generative model, we trained T5 to generate a class name.
As shown in our paper, the dataset construction process was different between Claim Detection Only and Joint Learning settings.

## Evaluation
The scripts for evaluation are put under evaluation dir.
There are two groups; one is scripts for each training and inference, and the other is for final evaluation.

Evaluation scripts in each training and inference.

- evaluate.py
    - This calculates metrics of each model's output.
- t5_predict_convert.py
    - This converts the T5 outputs into the same format as others.

Evaluation scripts for final models

- average_predictions.py
    - To make a final average model, this finds the best hyperparameter in each fold of each model.
    - This calculates the average output as the final ensemble model.
- output_test_predictions.py
    - This converts the final prediction into NTCIR-16 FinNum-3 format.