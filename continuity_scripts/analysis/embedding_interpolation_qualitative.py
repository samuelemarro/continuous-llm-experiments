import sys
sys.path.append('.')

import json

from transformers import AutoModelForCausalLM, AutoTokenizer

setups = [
    ('water_juice', 'drink_space', 'Is [] a drink?', 'water', 'juice')
]

interpolation_technique = 'linear'

model_configs = [
    ('meta-llama/Meta-Llama-3-8B'),
    ('meta-llama/Llama-2-13b-chat-hf'),
    ('google/gemma-7b'),
    ('google/gemma-2-9b'),
    ('microsoft/Phi-3-medium-4k-instruct'),
    ('mistralai/Mistral-7B-v0.3')
]

for model_name in model_configs:
    for i, setup in enumerate(setups):
        for experiment_category, name, question, word_a, word_b in setups:
            save_path = f'results/batched/embedding_interpolation/{experiment_category}/{name}/{i}/{interpolation_technique}/embedding_interpolation-batched-{model_name.lower().split("/")[1]}-{name}-{i}-{experiment_category}-{interpolation_technique}'

            try:
                with open(save_path + '.json') as f:
                    results = json.load(f)
            except:
                continue

            consistently_yes = True
            consistently_no = True
            for result_yes, result_no in zip(results['Yes'], results['No']):
                if result_no > result_yes:
                    consistently_yes = False
                if result_yes > result_no:
                    consistently_no = False
            print(consistently_yes, consistently_no)