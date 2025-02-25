import sys
sys.path.append('.')

import json
from pathlib import Path

from Levenshtein import distance

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.duration_shrink import run_counting_experiment

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

    for model_name in model_configs:
        model_name_short = model_name.lower().split("/")[1]
        curve_completeness = []
        curve_counterfactual_completeness = []
        curve_iou = []
        for threshold in np.linspace(0, 1, 101, endpoint=True):
            completeness_scores = []
            iou_scores = []
            counterfactual_completeness_scores = []

            for i, data in enumerate(setups):
                save_path = f'results/batched/counting_events/{i}/{model_name_short}/counting_events-{i}-{model_name_short}'

                try:
                    with open(save_path + '.json') as f:
                        results = json.load(f)
                except:
                    continue

                num_sentences = data['num_sentences']
                expected_peaks = set([int(i) for i in range(1,  num_sentences + 1)])
                found_peaks = set()
                found_conterfactual_peaks = set()
                unexpected_peaks = set([
                    int(i) for i in range(0, 10)
                    if i not in expected_peaks
                ])

                for peak in expected_peaks:
                    if str(peak) in results:
                        max_peak = max(results[str(peak)])

                        if max_peak >= threshold:
                            found_peaks.add(peak)
                        
                        last_peak = results[str(peak)][-1]
                        if last_peak >= threshold:
                            found_conterfactual_peaks.add(peak)
                for peak in unexpected_peaks:
                    if str(peak) in results:
                        max_peak = max(results[str(peak)])

                        if max_peak >= threshold:
                            found_peaks.add(peak)
                        
                        last_peak = results[str(peak)][-1]
                        if last_peak >= threshold:
                            found_conterfactual_peaks.add(peak)
                
                completeness_scores.append(len(found_peaks & expected_peaks) / len(expected_peaks))
                counterfactual_completeness_scores.append(len(found_conterfactual_peaks & expected_peaks) / len(expected_peaks))


                iou_scores.append(len(found_peaks & expected_peaks) / len(found_peaks | expected_peaks))
            #print(model_name, np.mean(completeness_scores), np.mean(purity_scores))
            curve_completeness.append(np.mean(completeness_scores))
            curve_counterfactual_completeness.append(np.mean(counterfactual_completeness_scores))
            #curve_purity.append(np.mean(purity_scores))
            curve_iou.append(np.mean(iou_scores))

            #print(model_name, threshold, np.mean(completeness_scores), np.mean(counterfactual_completeness_scores))
            #break
            #print('=======')
        plt.clf()
        #plt.plot(curve_completeness, curve_purity, label=f'{model_name} completeness')
        #plt.plot(np.linspace(0, 1, 100, endpoint=True), curve_purity, label=f'Purity')
        plt.plot(np.linspace(0, 1, 101, endpoint=True), curve_counterfactual_completeness, label=f'Counterfactual', color=colors[0])
        plt.plot(np.linspace(0, 1, 101, endpoint=True), curve_completeness, label=f'Observed', color=colors[1])
        #plt.plot(np.linspace(0, 1, 100, endpoint=True), curve_iou, label=f'IoU')
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Peak Threshold')
        plt.ylabel('Time Sensitivity')
        plt.tight_layout()
        path = Path(f'continuity_scripts/analysis/counting_events/graphs/completeness_purity_{model_name_short}.png')
        pdf_path = Path(f'continuity_scripts/analysis/counting_events/graphs/completeness_purity_{model_name_short}.pdf')
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(path))
        plt.savefig(str(pdf_path))

        completeness_auc = np.trapz(curve_completeness, dx=0.01)
        counterfactual_completeness_auc = np.trapz(curve_counterfactual_completeness, dx=0.01)

        print(model_name, len(completeness_scores), f'{counterfactual_completeness_auc:.4f}', f'{completeness_auc:.4f}')

        plt.clf()
        plt.plot(np.linspace(0, 1, 101, endpoint=True), curve_iou, label=f'IoU', color=colors[0])
        #plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Peak Threshold')
        plt.ylabel('IoU')
        plt.tight_layout()
        path = Path(f'continuity_scripts/analysis/counting_events/graphs/iou/iou_{model_name_short}.png')
        pdf_path = Path(f'continuity_scripts/analysis/counting_events/graphs/iou/iou_{model_name_short}.pdf')
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(path))
        plt.savefig(str(pdf_path))
