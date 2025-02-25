import sys
sys.path.append('.')

import json
from pathlib import Path

from Levenshtein import distance

from transformers import AutoModelForCausalLM, AutoTokenizer
#chosen_word = 'apple'
#category = 'fruit'

import numpy as np
import matplotlib.pyplot as plt


model_configs = [
    ('meta-llama/Meta-Llama-3-8B'),
    ('meta-llama/Llama-2-13b-chat-hf'),
    ('google/gemma-7b'),
    ('google/gemma-2-9b'),
    ('microsoft/Phi-3-medium-4k-instruct'),
    ('mistralai/Mistral-7B-v0.3')
]

#setups = [
#    #('shop', 'Question: Alice goes to the shop. She buys a carton of milk. She buys an apple. She buys a potato. She buys a loaf of bread. How many items did Alice buy? Reply with a single-digit number\nAnswer: ', 'She', '.'),
#    #('zoo', 'Question: the class went to the zoo. They saw a lion. They saw an elephant. They saw a giraffe. They saw a penguin. How many animals did the class see? Reply with a single-digit number\nAnswer: ', 'They', '.'),
#    #('beach', 'Question: Emily went to the beach. She found a seashell. She found a starfish. She found a smooth stone. She found a piece of seaweed. How many things did Emily find? Reply with a single-digit number\nAnswer: ', 'She', '.')
#    ('zoo', 'Alice goes to the shop. She buys a carton of milk. She buys an apple. She buys a potato. She buys a loaf of bread.', 'How many items did Alice buy?'),
#]

with open('continuity_scripts/data/counting_events_seq.json') as f:
    setups = json.load(f)

import seaborn as sns

style_info = plt.style.library['seaborn-v0_8-whitegrid']

style_info['font.family'] = "Times New Roman"
style_info['font.size'] = 14
print(style_info)

colors = sns.color_palette('husl', 5)

with plt.style.context(style_info):
    all_peak_rates = []
    all_counterfactual_peak_rates = []
    all_improvements = []

    for pure in [False, True]:

        if pure:
            print('====PURE====')
        else:
            print('====UNPURE====')

        for model_name in model_configs:
            model_name_short = model_name.lower().split("/")[1]

            peak_rates = []
            counterfactual_peak_rates = []
            improvements = []

            for i, data in enumerate(setups):
                save_path = f'results/batched/counting_events/{i}/{model_name_short}/counting_events-{i}-{model_name_short}'

                try:
                    with open(save_path + '.json') as f:
                        results = json.load(f)
                except:
                    print('Skipped', f'results/batched/counting_events/{i}/{model_name_short}/counting_events-{i}-{model_name_short}.json')
                    continue

                repetitions = data['num_sentences']
                expected_peaks = set([int(i) for i in range(1,  repetitions + 1)])
                found_peaks = set()

                for peak in expected_peaks if pure else range(10):
                    is_peak = False
                    for i in range(len(results[str(peak)])):
                        # Check if the value is above all other classes

                        value = results[str(peak)][i]
                        above_others = True

                        for other_peak in [str(x) for x in range(10) if x != peak]:
                            if value < results[str(other_peak)][i]:
                                above_others = False
                        if above_others:
                            is_peak = True
                            break
                    if is_peak:
                        found_peaks.add(peak)

                peak_rate = (len(found_peaks) / repetitions)
                counterfactual_peak_rate = (1 / repetitions)
                improvement = peak_rate / counterfactual_peak_rate

                peak_rates.append(peak_rate)
                all_peak_rates.append(peak_rate)

                counterfactual_peak_rates.append(counterfactual_peak_rate)
                all_counterfactual_peak_rates.append(counterfactual_peak_rate)

                improvements.append(improvement)
                all_improvements.append(improvement)

            print(f'{model_name}, {np.mean(peak_rates):.4}, {np.mean(counterfactual_peak_rates):.4}, {np.mean(improvements):.3}')

        print(f'Global {np.mean(all_peak_rates):.4}, {np.mean(all_counterfactual_peak_rates):.4}, {np.mean(all_improvements):.3}')
