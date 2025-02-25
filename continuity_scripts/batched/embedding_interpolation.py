import sys
sys.path.append('.')

import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.embedding_interpolation import run_embedding_interpolation_experiment



model_configs = [
    ('meta-llama/Meta-Llama-3-8B'),
    ('meta-llama/Llama-2-13b-chat-hf'),
    ('google/gemma-7b'),
    ('google/gemma-2-9b'),
    ('microsoft/Phi-3-medium-4k-instruct'),
    ('mistralai/Mistral-7B-v0.3')
]

#setups = [
#    ('water_juice', 'drink_space', 'Is [] a drink?', 'water', 'juice')
#]

with open('continuity_scripts/data/embedding_filtered.json', 'r') as f:
    setups = json.load(f)

setups = setups[:50]

print(len(setups))


interpolation_technique = 'linear'

for model_name in model_configs:

    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(len(setups))

    for i, setup in enumerate(setups):
        keys = list(setup['questions'].keys())
        questions = list(setup['questions'].values())
        assert len(keys) == len(questions)
        for j, (key, question) in enumerate(zip(keys, questions)):
            target_a = 'Yes'
            prefix_a = 'yes'
            target_b = 'No'
            prefix_b = 'no'

            sentence = f'Question: {question} (yes/no)\nAnswer:'
            word_a = setup['options'][0]
            word_b = setup['options'][1]
            sentence_a = sentence.replace('{}', word_a)
            sentence_b = sentence.replace('{}', word_b)

            print(sentence_a)
            print(sentence_b)

            interesting_outputs = {
                target_a: [prefix_a],
                target_b: [prefix_b]
            }
            #for extra_target, extra_prefixes in extra_targets:
            #    interesting_outputs[extra_target] = extra_prefixes

            model_name_short = model_name.lower().split("/")[1]
            save_path = f'results/batched/embedding_interpolation/{i}/{key}/{model_name_short}/embedding_interpolation-{i}-{key}-{model_name_short}'
            #save_path = f'results/batched/embedding_interpolation/{i}_{word_a}_{word_b}/{j}/{interpolation_technique}/embedding_interpolation-batched-{model_name.lower().split("/")[1]}-{word_a}_{word_b}_{i}_{j}_{interpolation_technique}'

            if Path(save_path + '.json').exists():
                print(f'Skipping {save_path}')
                continue

            try:
                run_embedding_interpolation_experiment(current_model, current_tokenizer, sentence_a, sentence_b, interesting_outputs, interpolation_technique, save_path, 'Interpolation Factor', legend=True, num_samples=40)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(e)

                failed_info = {
                    'model': model_name,
                    'index': i,
                    'key': key,
                    'question': question,
                    'error': str(e)
                }

                try:
                    with open('continuity_scripts/batched/embedding_interpolation_failed.json') as f:
                        current_failed = json.load(f)
                except:
                    current_failed = []

                current_failed.append(failed_info)

                with open('continuity_scripts/batched/embedding_interpolation_failed.json', 'w') as f:
                    json.dump(current_failed, f, indent=4)