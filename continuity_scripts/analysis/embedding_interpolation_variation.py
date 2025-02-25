import sys
sys.path.append('.')

import json

import numpy as np


def estimate_derivative(x, y):
    """
    Estimate the derivative dy/dx at each point using finite differences.

    Parameters:
        x (list of float): x coordinates.
        y (list of float): y coordinates, corresponding to x.

    Returns:
        list of float: derivative estimates at each x value.
    """
    if len(x) != len(y):
        raise ValueError("The lists x and y must have the same length.")

    n = len(x)
    if n < 2:
        raise ValueError("At least two points are required to compute a derivative.")

    dydx = [0] * n

    # Forward difference for the first point
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])

    # Central difference for the interior points
    for i in range(1, n - 1):
        dx = x[i + 1] - x[i - 1]
        if dx == 0:
            raise ValueError(f"Zero division error: x[{i+1}] and x[{i-1}] are equal.")
        dydx[i] = (y[i + 1] - y[i - 1]) / dx

    # Backward difference for the last point
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

    return dydx


with open('continuity_scripts/data/embedding_filtered.json', 'r') as f:
    setups = json.load(f)

setups = setups[:50]

interpolation_technique = 'linear'

model_configs = [
    ('meta-llama/Meta-Llama-3-8B'),
    ('meta-llama/Llama-2-13b-chat-hf'),
    ('google/gemma-7b'),
    ('google/gemma-2-9b'),
    ('microsoft/Phi-3-medium-4k-instruct'),
    ('mistralai/Mistral-7B-v0.3')
]

num_samples = 40
x = np.linspace(0, 1, num_samples, endpoint=True)


lipschitz_estimates = {
    model_name: {
        k: { 'Yes': [], 'No': [] } for k in ['neither', 'one', 'both']
    }
    for model_name in model_configs
}
variations = {
    model_name: []
    for model_name in model_configs
}
has_variations = {
    model_name: []
    for model_name in model_configs
}

total_records = 0
total_valid = 0

for model_name in model_configs:
    num_records = 0
    num_valid = 0
    for i, setup in enumerate(setups):
        for data in setups:
            keys = list(data['questions'].keys())
            questions = list(data['questions'].values())
            for j, (key, question) in enumerate(zip(keys, questions)):
                model_name_short = model_name.lower().split("/")[1]
                save_path = f'results/batched/embedding_interpolation/{i}/{key}/{model_name_short}/embedding_interpolation-{i}-{key}-{model_name_short}'
            
                num_records += 1
                try:
                    with open(save_path + '.json') as f:
                        results = json.load(f)
                    num_valid += 1
                except:
                    #print('Skipped', model_name, i)
                    continue

                local_variations = []

                for answer in ['Yes', 'No']:

                    values = results[answer]

                    initial_value = values[0]
                    final_value = values[-1]

                    low = min(initial_value, final_value)
                    high = max(initial_value, final_value)

                    out_of_bounds = []


                    for value in values[1:-1]:
                        if value < low:
                            out_of_bounds.append(np.abs(low - value))
                        elif value > high:
                            out_of_bounds.append(np.abs(high - value))
                    
                    if len(out_of_bounds) > 0:
                        local_variation = np.max(out_of_bounds)
                    else:
                        local_variation = 0

                    local_variations.append(local_variation)
                    #print(variation)

                max_variation = np.max(local_variations)
                variations[model_name].append(max_variation)
                has_variations[model_name].append(1 if max_variation >= 0.05 else 0)

    print(model_name, num_valid, num_records, num_valid / num_records)

    total_records += num_records
    total_valid += num_valid

print('Global', total_valid, total_records, total_valid / total_records)

#for model_name in model_configs:
#    for k in ['neither', 'one', 'both']:
#        for answer in ['Yes', 'No']:
#            lipschitz_estimates[model_name][k][answer] = np.mean(lipschitz_estimates[model_name][k][answer])

#print(lipschitz_estimates)

import seaborn as sns
import matplotlib.pyplot as plt

style_info = plt.style.library['seaborn-v0_8-whitegrid']

style_info['font.family'] = "Times New Roman"
style_info['font.size'] = 14

from pathlib import Path

with plt.style.context(style_info):

    for model_name in model_configs:
        #print(np.count_nonzero(has_variations[model_name]))
        #print(np.mean(np.array(has_variations[model_name]).astype(float)))
        #print(len(np.unique(variations[model_name])))
        print(f'{model_name} {np.mean(variations[model_name]):.4f} {np.mean(has_variations[model_name])*100:.2f}%')
        plt.clf()
        sns.histplot(variations[model_name], bins=10, kde=False, label=model_name)
        plt.xlabel('Variation')
        plt.ylabel('Frequency')

        save_path = f'continuity_scripts/analysis/embedding_interpolation/variation/embedding_interpolation-variation-{model_name.lower().split("/")[1]}'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path + '.png')
        plt.savefig(save_path + '.pdf')

    # Merge all the variations
    all_variations = []
    all_has_variations = []

    for model_name in model_configs:
        all_variations += variations[model_name]
        all_has_variations += has_variations[model_name]

    print(f'Global {np.mean(all_variations):.4f} {np.mean(all_has_variations)*100:.2f}%')