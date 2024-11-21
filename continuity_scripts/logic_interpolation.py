import sys
sys.path.append('.')

import itertools

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.embedding_interpolation import run_embedding_interpolation_experiment



model_configs = [
    ('meta-llama/Meta-Llama-3-8B', 'standard'),
    ('meta-llama/Llama-2-13b-chat-hf', 'variant1'),
    ('google/gemma-7b', 'variant1'),
    ('google/gemma-2-9b', 'standard'),
    ('microsoft/Phi-3-medium-4k-instruct', 'standard'),
    ('mistralai/Mistral-7B-v0.3', 'variant1')
]


interpolation_technique = 'linear'

operators = ['AND', 'OR', 'NAND', 'NOR', 'XOR']

for model_name, prompt_type in model_configs:

    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for operator_subset in itertools.combinations(operators, 2):
        operator_subset = sorted(list(operator_subset))
        operator_a = operator_subset[0]
        operator_b = operator_subset[1]

        for value_left, value_right in [('0', '0'), ('0', '1'), ('1', '0'), ('1', '1')]:

            sentence = f'Question: What is the truth value of the following statement?\n{value_left} [] {value_right}\nAnswer (0 or 1): '

            sentence_a = sentence.replace('[]', operator_a)
            sentence_b = sentence.replace('[]', operator_b)

            print(sentence_a)
            print(sentence_b)

            interesting_outputs = {
                'True': ['1', 'tr'],
                'False': ['0', 'fa'] 
            }

            #save_path = f'results/embedding_interpolation/{experiment_category}/{name}/{interpolation_technique}/embedding_interpolation-{model_name.lower().split("/")[1]}-{name}-{experiment_category}-{interpolation_technique}'
            
            model_name_simplified = model_name.lower().split("/")[1]
            save_path = f'results/logic_interpolation/{model_name_simplified}/{operator_a}_{operator_b}/{value_left}_{value_right}/logic_interpolation-{model_name_simplified}-{operator_a}_{operator_b}-{value_left}_{value_right}'

            try:
                run_embedding_interpolation_experiment(current_model, current_tokenizer, sentence_a, sentence_b, interesting_outputs, interpolation_technique, save_path, 'Interpolation Factor', legend=True)
            except Exception as e:
                print(e)
                continue
    #run_experiment(current_model, current_tokenizer, sentence, shrink_start, shrink_end, interesting_outputs, save_path)