import sys
sys.path.append('.')

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.duration_shrink import run_counting_experiment

import numpy as np


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

total_num_records = 0
total_num_valid = 0
total_num_successful = 0
total_num_strongly_successful = 0
total_num_completely_successful = 0

for model_name in model_configs:
    num_records = 0
    num_valid = 0
    num_successful = 0
    num_strongly_successful = 0
    num_completely_successful = 0

    for i, data in enumerate(dataset):
        template = data['template']

        # Find the first word after {item_1}
        first_word_after_item_1 = template.split('{item_1}')[1].split(' ')[1]

        if data['shrink_1']:
            select_start = None
            select_end = first_word_after_item_1
            shrunk_number = data['fields']['item_1']
            unshrunk_number = data['fields']['item_2']
        else:
            select_start = first_word_after_item_1
            select_end = None
            shrunk_number = data['fields']['item_2']
            unshrunk_number = data['fields']['item_1']


        sentence = template.format(**data['fields'])

        first_digit_shrunk = int(str(shrunk_number)[0])
        second_digit_shrunk = int(str(shrunk_number)[1])

        unshrunk_output = str(unshrunk_number + shrunk_number)[0]

        shrunk_outputs = list(set([
            str(unshrunk_number + first_digit_shrunk)[0],
            str(unshrunk_number + second_digit_shrunk)[0]
        ]))

        #print({
        #    'sentence': sentence,
        #    'first_digit_shrunk': first_digit_shrunk,
        #    'second_digit_shrunk': second_digit_shrunk,
        #    'unshrunk_output': unshrunk_output,
        #    'shrunk_outputs': shrunk_outputs
        #})

        other_outputs = list(set([
            str(unshrunk_number + x)[0] for x in range(10)
        ]) - set(shrunk_outputs) - set([unshrunk_output]))

        model_name_short = model_name.lower().split("/")[1]
        save_path = f'results/batched/sums/{i}/{model_name_short}/sum-{i}-{model_name_short}.json'

        import pathlib

        if not pathlib.Path(save_path).exists():
            print('Skipping', save_path)
            continue

        with open(save_path, 'r') as f:
            results = json.load(f)

        num_records += 1

        unshrunk_probabilities = np.array(results[unshrunk_output])

        # Sum the outputs for the shrunk digits
        shrunk_probabilities = np.zeros(len(unshrunk_probabilities))

        for shrunk_output in shrunk_outputs:
            shrunk_probabilities += np.array(results[shrunk_output])

        # Consider the other outputs individually
        other_probabilities = [
            np.array(results[other_output]) for other_output in other_outputs
        ]

        # An experiment is valid if the last unshrunk output is the most probable
        valid = unshrunk_probabilities[-1] > shrunk_probabilities[-1] and all(
            unshrunk_probabilities[-1] > other_probability[-1] for other_probability in other_probabilities
        )

        #valid = valid and unshrunk_probabilities[-1] > sum([other_probability[-1] for other_probability in other_probabilities] + [shrunk_probabilities[-1]])

        if valid:
            num_valid += 1
        else:
            continue

        successful = False
        # An experiment is successful if at any point the shrunk output is more probable than the unshrunk output
        for i in range(len(shrunk_probabilities)):
            if shrunk_probabilities[i] > unshrunk_probabilities[i]:
                successful = True
                break

        if successful:
            num_successful += 1
        else:
            continue

        # An experiment is strongly successful if at any point the shrunk output is more probable than the unshrunk output and all other outputs
        strongly_successful = False

        for i in range(len(shrunk_probabilities)):
            if shrunk_probabilities[i] > unshrunk_probabilities[i] and all(
                shrunk_probabilities[i] > other_probability[i] for other_probability in other_probabilities
            ):
                strongly_successful = True
                break
        
        if strongly_successful:
            num_strongly_successful += 1
        else:
            continue

        # An experiment is completely successful if all the previous conditions are met and at every point no other output (aside from the shrunk output) is more probable than the unshrunk output

        completely_successful = True

        for i in range(len(shrunk_probabilities)):
            if shrunk_probabilities[i] > unshrunk_probabilities[i] and any(
                shrunk_probabilities[i] < other_probability[i] for other_probability in other_probabilities
            ):
                completely_successful = False
                break

        if completely_successful:
            num_completely_successful += 1

    #print(f'{model_name}: {num_records} records, {num_valid} valid ({num_valid / num_records * 100:.2f}%)')

    total_num_records += num_records
    total_num_valid += num_valid
    total_num_successful += num_successful
    total_num_strongly_successful += num_strongly_successful
    total_num_completely_successful += num_completely_successful

    if num_valid > 0:
        #print(f'{model_name}: {num_successful} successful ({num_successful / num_valid * 100:.2f}%)')
        #print(f'{model_name}: {num_strongly_successful} strongly successful ({num_strongly_successful / num_valid * 100:.2f}%)')
        #print(f'{model_name}: {num_completely_successful} completely successful ({num_completely_successful / num_valid * 100:.2f}%)')

        success_rate = num_successful / num_valid
        strong_success_rate = num_strongly_successful / num_valid
        complete_success_rate = num_completely_successful / num_valid

        print(f'{model_name} {success_rate * 100:.2f}% {strong_success_rate * 100:.2f}% {complete_success_rate * 100:.2f}%')
    #print()
        
print(f'Total: {total_num_records} records, {total_num_valid} valid ({total_num_valid / total_num_records * 100:.2f}%)')
print(f'Total: {total_num_successful} successful ({total_num_successful / total_num_valid * 100:.2f}%)')
print(f'Total: {total_num_strongly_successful} strongly successful ({total_num_strongly_successful / total_num_valid * 100:.2f}%)')
print(f'Total: {total_num_completely_successful} completely successful ({total_num_completely_successful / total_num_valid * 100:.2f}%)')
print(f'Global {total_num_successful / total_num_valid * 100:.2f}% {total_num_strongly_successful / total_num_valid * 100:.2f}% {total_num_completely_successful / total_num_valid * 100:.2f}%')