# gpt3_simple_primer

Simple GPT-3 primer using `openai`.

## Background

Generative Pre-trained Transformer 3 (GPT-3) is an autoregressive language model that uses deep learning to produce human-like text. For more information, visit https://openai.com/blog/openai-api/.

The [OpenAI Python library](https://github.com/openai/openai-python) is the official Python wrapper for the OpenAI API. The purpose of this library is to simplify the priming process by providing easy to use methods for setting the instructions and adding examples.

## Priming

Priming is the practice of providing an initial prompt to the language model to improve subsequent model predictions.

GPT-3 generally does very well even with short instructions and a few examples of your intended use case. Examples are typically delimited based on input and output. For instance, GPT-3 can be used to predict food ingredients based on the following prompt:

```
Given the name of a food, list the ingredients used to make this meal.

Food: apple pie
Ingredients: apple, butter, flour, egg, cinnamon, crust, sugar

Food: guacamole
Ingredients: avocado, tomato, onion, lime, salt
```

## Requirements

You will need an API key from OpenAI to access GPT-3.

## Usage

`input_text` and `output_text` determines how input and output are delimited in the examples. The default is to use `Input` and `Output`.

```
from gpt3_simple_primer import GPT3Generator

key = 'sk-xxxxx'

generator = GPT3Generator(engine='davinci',
                          max_tokens=20,
                          temperature=0.5,
                          top_p=1,
                          input_text='Food',
                          output_text='Ingredients')

generator.set_key(key)
generator.set_instructions('List the ingredients for this meal.')
generator.add_example('apple pie', 'apple, butter, flour, egg, cinnamon, crust, sugar')
generator.add_example('guacamole', 'avocado, tomato, onion, lime, salt')

# key lime, egg, sugar, butter, graham cracker, cream
generator.generate('key lime pie')
```

To see the prompt used for priming:

```
generator.get_prompt()
```

To remove an example from the prompt:

```
generator.remove_example('apple pie')
```