# Prompt Tuning
This is the pytorch implementation of [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691v2).

Currently, we support the following huggigface models:
- `GPT2LMModel`

## Usage
See `example.ipynb` for more details. 
```python
from model import GPT2PromptTuningLM

# number of prompt tokens
n_prompt_tokens = 20
# If True, soft prompt will be initialized from vocab 
# Otherwise, you can set `random_range` to initialize by randomization.
init_from_vocab = True
# random_range = 0.5

# Initialize GPT2LM with soft prompt
model = GPT2PromptTuningLM.from_pretrained(
    "gpt2",
    n_tokens=n_prompt_tokens,
    initialize_from_vocab=init_from_vocab
)
```


## Reference
- https://github.com/corolla-johnson/mkultra
- https://github.com/kipgparker/soft-prompt-tuning