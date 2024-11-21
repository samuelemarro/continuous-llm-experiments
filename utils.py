from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

DIGITS = [str(i) for i in range(10)]


# Use in the rainbow order
COLORS = ['tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:cyan', 'tab:blue', 'tab:pink', 'tab:purple', 'tab:brown', 'xkcd:burnt siena']
COLORS = {k : v for k, v in zip(DIGITS, COLORS)}
COLORS['other_digits'] = 'tab:gray'
COLORS['other_tokens'] = (26 / 255, 25 / 255, 25 / 255)

def append_results_info(probs, result):
    assert set(DIGITS) == set(probs.keys())

    # For some God-forsaken reason, there are multiple results for the same digit.
    # This is probably due to equivalent Unicode characters. The impact is minimal
    # (on the scale of 1e-12), but for completeness we sum them up

    for digit in DIGITS:
        total = 0
        for key, token_idx, value in result:
            if key == digit:
                total += value
        assert total != 0
        probs[digit].append(total)

def get_tracked_tokens(vocabulary_dict, interesting_outputs):
    tracked_tokens = { x: [] for x in interesting_outputs }

    for k, v in vocabulary_dict.items():
        if '0x' in k:
            # Byte pair encoding token, skip
            continue
        for category_name, allowed_outputs in interesting_outputs.items():
            if isinstance(allowed_outputs, str):
                allowed_outputs = [allowed_outputs]
            
            found = False
            for allowed_output in allowed_outputs:
                if allowed_output.lower() in k.lower():
                    tracked_tokens[category_name].append(v)
                    #interesting_tokens.append((k, v))
                    found = True
                    break
            if found:
                break
    
    return tracked_tokens

def add_results(results, tracked_tokens, probs):
    for interesting_output, interesting_tokens in tracked_tokens.items():
        total_probability = 0
        for token in interesting_tokens:
            total_probability += probs[0, -1, token].item()
        results[interesting_output].append(total_probability)

COLUMNS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 3,
    6: 3,
    7: 4,
    8: 4,
    9: 4,
    10: 4,
    11: 4,
    12: 4,
}

def reorder(names, num_columns):
    return (sum((name_list[i::num_columns] for i in range(num_columns)),[]) for name_list in names)

def plot_results(interpolation_factors, all_results, save_path, xlabel, legend=False):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    total_probabilities = np.zeros(len(interpolation_factors))
                                   
    style_info = plt.style.library['seaborn-v0_8-whitegrid']

    style_info['font.family'] = "Times New Roman"
    style_info['font.size'] = 14
    print(style_info)

    colors = sns.color_palette('Set2') + sns.color_palette('husl', 5)

    with plt.style.context(style_info):
        figure = plt.figure()
        ax = figure.add_subplot(111)

        plt.grid(True, alpha=0.8)

        for i, (category_name, probabilities) in enumerate(all_results.items()):
            if category_name in DIGITS:
                color_index = int(category_name)
            else:
                color_index = i % len(colors)
            #print(category_name, probabilities)

            total_probabilities += np.array(probabilities)

            if 'other' in category_name.lower():
                ax.plot(interpolation_factors, probabilities, label=category_name, c='0.4', linestyle='--')
            else:
                ax.plot(interpolation_factors, probabilities, c=colors[color_index], label=category_name)

        ax.plot(interpolation_factors, 1 - total_probabilities, label='Other Tokens', c='0.2', linestyle='dotted')

        plt.xlabel(xlabel, weight='bold')
        plt.ylabel('Probability', weight='bold')

        plt.xlim([min(interpolation_factors), max(interpolation_factors)])

        plt.savefig(save_path + '.png', bbox_inches='tight', format='png')
        plt.savefig(save_path + '.pdf', bbox_inches='tight', format='pdf')

        if legend:
            h_l = plt.gca().get_legend_handles_labels()
            num_columns = COLUMNS[len(all_results) + 1]

            lgd = plt.legend(*reorder(h_l, num_columns), loc='upper center', bbox_to_anchor=(0.5, -0.125), ncol=num_columns, frameon=True)
            extra_artists = (lgd,)

            plt.savefig(save_path + '-legend.png', bbox_inches='tight', bbox_extra_artists=extra_artists, format='png')
            plt.savefig(save_path + '-legend.pdf', bbox_inches='tight', bbox_extra_artists=extra_artists, format='pdf')
        
        plt.clf()