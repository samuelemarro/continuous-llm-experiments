import sys
sys.path.append('.')

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.duration_shrink import run_counting_experiment

repetitions = 4

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
    ('shop', 'Question: Alice goes to the shop. She buys a carton of milk. She buys an apple. She buys a potato. She buys a loaf of bread. How many items did Alice buy? Reply with a single-digit number\nAnswer: ', 'She', '.'),
    ('zoo', 'Question: the class went to the zoo. They saw a lion. They saw an elephant. They saw a giraffe. They saw a penguin. How many animals did the class see? Reply with a single-digit number\nAnswer: ', 'They', '.'),
    ('beach', 'Question: Emily went to the beach. She found a seashell. She found a starfish. She found a smooth stone. She found a piece of seaweed. How many things did Emily find? Reply with a single-digit number\nAnswer: ', 'She', '.')
]

for model_name, prompt_type in model_configs:

    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for setup_name, sentence, shrink_start, shrink_end in setups:
        save_path = f'results/shrink/counting_events/{setup_name}/shrink-{model_name.lower().split("/")[1]}-{setup_name}-counting_events'

        run_counting_experiment(current_model, current_tokenizer, sentence, None, shrink_start, shrink_end, interesting_outputs, save_path, 'Duration Factor', True)
