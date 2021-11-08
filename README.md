# gpt3_simple_primer

Wrapper used to simplify GPT-3 priming.

## Background

Generative Pre-trained Transformer 3 (GPT-3) is an autoregressive language model that uses deep learning to produce human-like text. For more information, visit https://openai.com/blog/openai-api/.

**Priming:** the initial prompt fed to the language model for subsequent text generation

The [OpenAI Python library](https://github.com/openai/openai-python) is the official Python wrapper for the OpenAI API. The purpose of this library is to simplify the priming process by providing easy to use methods for setting the instructions and adding examples.

## Requirements

You will need an API key from OpenAI to access GPT-3.

## Usage

```
from gpt3_simple_primer import GPT3Generator

key = 'sk-xxxxx'

generator = GPT3Generator(engine='davinci',
                          max_tokens=20,
                          temperature=0.5,
                          top_p=1)

generator.set_key(key)
generator.set_instructions('List the ingredients for this meal.')
generator.add_example('apple pie', 'apple, butter, flour, egg, cinnamon, crust, sugar')
generator.add_example('guacamole', 'avocado, tomato, onion, lime, salt')

# key lime, egg, sugar, butter, graham cracker, cream
generator.generate('key lime pie')
```
