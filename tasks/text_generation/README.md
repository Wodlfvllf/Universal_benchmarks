# Text Generation Task

This task handles open-ended text generation problems. The model is given a prompt and is expected to generate a continuation or completion.

This is a foundational task type that can be inherited by more specific generation tasks like Summarization or Translation.

## Features

- Takes a text prompt as input.
- Generates a text completion.
- Computes standard generation metrics like ROUGE and BLEU.

## Configuration

When using this task in a benchmark `config.yaml`, the `task_type` should be set to `text_generation`.

- `input_columns`: A list containing the column name for the input prompt.
- `label_column`: The name of the column containing the reference completion.
