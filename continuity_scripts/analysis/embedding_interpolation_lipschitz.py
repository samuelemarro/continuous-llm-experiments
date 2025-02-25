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
global_lipschitz_estimates = {
    model_name: []
    for model_name in model_configs
}

for model_name in model_configs:
    for i, setup in enumerate(setups):
        for data in setups:
            keys = list(data['questions'].keys())
            questions = list(data['questions'].values())
            for j, (key, question) in enumerate(zip(keys, questions)):
                model_name_short = model_name.lower().split("/")[1]
                save_path = f'results/batched/embedding_interpolation/{i}/{key}/{model_name_short}/embedding_interpolation-{i}-{key}-{model_name_short}'
            
                try:
                    with open(save_path + '.json') as f:
                        results = json.load(f)
                except:
                    print('Skipped', model_name, i)
                    continue

                for answer in ['Yes', 'No']:
                    normalization_constant = np.max(results[answer]) - np.min(results[answer])
                    lipschitz_estimate = np.max(np.abs(estimate_derivative(x, results[answer]))) / normalization_constant

                    if key == 'first' or key == 'second':
                        lipschitz_estimates[model_name]['one'][answer].append(lipschitz_estimate)
                    else:
                        lipschitz_estimates[model_name][key][answer].append(lipschitz_estimate)

                    global_lipschitz_estimates[model_name].append(lipschitz_estimate)

total_estimates = []

for model_name in model_configs:
    for k in ['neither', 'one', 'both']:
        for answer in ['Yes', 'No']:
            total_estimates += lipschitz_estimates[model_name][k][answer]
            lipschitz_estimates[model_name][k][answer] = np.mean(lipschitz_estimates[model_name][k][answer])

print(lipschitz_estimates)

for model_name in model_configs:
    print(f'{model_name} {np.mean(global_lipschitz_estimates[model_name]):.3f}')
print(f'Global {np.mean(total_estimates):.3f}')