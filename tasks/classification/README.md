# Text Classification Task

This task handles text classification problems, where a model is given one or more text inputs and must assign a label from a predefined set.

## Features

- Supports single-text and text-pair classification.
- Automatically encodes string labels into integers.
- Can be used for binary, multi-class, and multi-label (in the future) classification.
- Computes standard metrics like Accuracy, F1-score, Precision, and Recall.

## Configuration

When using this task in a benchmark `config.yaml`, the `task_type` should be set to `classification`.

- `input_columns`: A list of one or two column names from the dataset to be used as input text.
- `label_column`: The name of the column containing the ground truth labels.
