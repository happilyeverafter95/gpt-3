# gpt-3

Python wrapper for GPT-3.

## Background

Generative Pre-trained Transformer 3 (GPT-3) is an autoregressive language model that uses deep learning to produce human-like text. For more information, visit https://openai.com/blog/openai-api/.

## Requirements

You will need an API key from OpenAI to access GPT-3.

## Usage

```
from gpt3_wrapper import GPT3Generator

key = 'sk-xxxxx'

generator = GPT3Generator(engine='davinci',
                          max_tokens=20,
                          temperature=0.5,
                          top_p=1)

generator.set_key(key)
generator.set_instructions('List the ingredients for this meal.')
generator.add_example('apple pie', 'apple, butter, flour, egg, cinnamon, crust, sugar')
generator.add_example('guacamole', 'avocado, tomato, onion, lime, salt')

# This should return 'graham cracker, lime, milk, sugar, egg, vanilla'
generator.generate('key lime pie')
```
