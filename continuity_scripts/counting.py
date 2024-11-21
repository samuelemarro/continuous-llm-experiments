import sys
sys.path.append('.')

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.duration_shrink import run_counting_experiment

#chosen_word = 'apple'
#category = 'fruit'
repetitions = 4


# Llama 3 8b <---
# Llama 3 70b (only if it fits in memory)
# Llama 2 7b
# Llama 2 13b <---
# Llama 2 70b (only if it fits in memory)
# Phi 3 mini (2b)
# Phi 3 small (7b)
# Phi 3 medium (14b) <---
# Gemma 1 2b
# Gemma 1 7b <---
# Gemma 2 2b
# Gemma 2 9b <---
# Gemma 2 27b (only if it fits in memory)
# Mistral 7b <---

# Llama 2&3, Phi 3, Gemma 1&2, Mistral
# Llama 2 7/13/70b, Llama 3 8/70b, Phi 3 mini/small/medium, Gemma 1 2/7b, Gemma 2 2/9/27b, Mistral 7b


#sentence = f'Question: In the sentence "{" ".join([chosen_word] * repetitions)}", how many times is a {category} mentioned? Reply with a single-digit number\nAnswer: '
#
#variant_sentence = f'Question: How many {category}s are listed in the sentence "{" ".join([chosen_word] * repetitions)}"? Reply with a single-digit number\nAnswer: '
# Mistral

#sentence = f'Question: How many {category} are mentioned in the sentence "{" ".join([chosen_word] * repetitions)}"? Reply with a single-digit number\nAnswer: '


interesting_outputs = {}

for i in range(0, repetitions + 1):
    interesting_outputs[str(i)] = str(i)

interesting_outputs['Other Digits'] = [str(i) for i in range(10) if str(i) not in interesting_outputs]

print(interesting_outputs)

model_configs = [
    ('meta-llama/Meta-Llama-3-8B', 'standard'),
    ('meta-llama/Llama-2-13b-chat-hf', 'variant1'),
    ('google/gemma-7b', 'variant1'),
    ('google/gemma-2-9b', 'standard'),
    ('microsoft/Phi-3-medium-4k-instruct', 'standard'),
    ('mistralai/Mistral-7B-v0.3', 'variant1')
]

setups = [
    ('apple', 'fruit', 'a fruit'),
    ('cat', 'animal', 'an animal'),
    ('rose', 'flower', 'a flower')
]

for model_name, prompt_type in model_configs:

    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for word, category, category_with_article in setups:
        sentence = f'Question: In the sentence "{" ".join([word] * repetitions)}", how many times is {category_with_article} mentioned? Reply with a single-digit number\nAnswer: '
        variant_sentence = f'Question: How many {category}s are listed in the sentence "{" ".join([word] * repetitions)}"? Reply with a single-digit number\nAnswer: '

        save_path = f'results/shrink/counting/{word}/shrink-{model_name.lower().split("/")[1]}-{word}-counting'

        shrink_start = word
        shrink_end = word

        used_sentence = sentence if prompt_type == 'standard' else variant_sentence

        run_counting_experiment(current_model, current_tokenizer, used_sentence, None, shrink_start, shrink_end, interesting_outputs, save_path, 'Duration Factor', legend=True)