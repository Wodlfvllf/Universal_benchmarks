# Multiple Choice Question Answering Task

This task handles multiple choice question answering problems. The model is given a context, a question, and a list of possible choices, and it must select the correct choice.

## Features

- Supports questions with a variable number of choices.
- Can handle an optional context string.
- Formats the input into a clear prompt for the model.
- Computes accuracy as the primary metric.

## Configuration

When using this task in a benchmark `config.yaml`, the `task_type` should be set to `multiple_choice`.

- `input_columns`: A list of column names from the dataset. The task expects to find columns for the question and the choices.
- `label_column`: The name of the column containing the index of the correct answer.
