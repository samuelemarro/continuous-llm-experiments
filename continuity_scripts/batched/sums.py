import sys
sys.path.append('.')

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.duration_shrink import run_counting_experiment


model_configs = [
    ('meta-llama/Meta-Llama-3-8B'),
    ('meta-llama/Llama-2-13b-chat-hf'),
    ('google/gemma-7b'),
    ('google/gemma-2-9b'),
    ('microsoft/Phi-3-medium-4k-instruct'),
    ('mistralai/Mistral-7B-v0.3')
]

#setups = [
#    (24, 13),
#    (32, 56),
#    (13, 74)
#]

current_model = None
current_tokenizer = None

#dataset = [
#    'The sum of {number_1} and {number_2} is ',
#]

with open('continuity_scripts/data/sums.json') as f:
    dataset = json.load(f)

NUM_ELEMENTS = 200
dataset = dataset[:NUM_ELEMENTS]

for model_name in model_configs:
    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, data in enumerate(dataset):
        #print(data)
        template = data['template']

        # Find the first word after {item_1}
        first_word_after_item_1 = template.split('{item_1}')[1].split(' ')[1]

        if data['shrink_1']:
            select_start = None
            select_end = 'and'
            shrunk_number = data['fields']['item_1']
            unshrunk_number = data['fields']['item_2']
        else:
            select_start = 'and'
            select_end = '.'
            shrunk_number = data['fields']['item_2']
            unshrunk_number = data['fields']['item_1']


        sentence = template.format(**data['fields'])

        first_digit_shrunk = int(str(shrunk_number)[0])
        second_digit_shrunk = int(str(shrunk_number)[1])

        #interesting_digits = [
        #    str(unshrunk_number + shrunk_number)[0],
        #    str(unshrunk_number + first_digit_shrunk)[0],
        #    str(unshrunk_number + second_digit_shrunk)[0]
        #]

        # Track every digit from 0 to 9. We'll count them after the experiment.

        interesting_digits = [str(i) for i in range(10)]

        interesting_outputs = {i : i for i in interesting_digits}

        model_name_short = model_name.lower().split("/")[1]
        save_path = f'results/batched/sums/{i}/{model_name_short}/sum-{i}-{model_name_short}'

        shrink_start = str(first_digit_shrunk)
        shrink_end = str(second_digit_shrunk)

        from pathlib import Path
        if Path(save_path + '.json').exists():
            continue

        #print({
        #    'shrunk_number': shrunk_number,
        #    'unshrunk_number': unshrunk_number,
        #    'first_digit_shrunk': first_digit_shrunk,
        #    'second_digit_shrunk': second_digit_shrunk,
        #    'shrink_start': shrink_start,
        #    'shrink_end': shrink_end
        #})
        #assert False

        try:
            run_counting_experiment(current_model, current_tokenizer, sentence, select_end, shrink_start, shrink_end, interesting_outputs, save_path, 'Duration Factor', True, select_start=select_start)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            failed_info = {
                'model': model_name,
                'sentence': sentence,
                'select_start': select_start,
                'select_end': select_end,
                'shrink_start': shrink_start,
                'shrink_end': shrink_end,
                'interesting_outputs': interesting_outputs,
                'save_path': save_path,
                'cause': str(e)
            }

            try:
                with open('continuity_scripts/batched/sums_failed.json') as f:
                    current_failed = json.load(f)
            except:
                current_failed = []

            current_failed.append(failed_info)

            with open('continuity_scripts/batched/sums_failed.json', 'w') as f:
                json.dump(current_failed, f, indent=4)

