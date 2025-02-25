import sys
sys.path.append('.')

import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.duration_shrink import run_counting_experiment

#chosen_word = 'apple'
#category = 'fruit'


model_configs = [
    ('meta-llama/Meta-Llama-3-8B'),
    ('meta-llama/Llama-2-13b-chat-hf'),
    ('google/gemma-7b'),
    ('google/gemma-2-9b'),
    ('microsoft/Phi-3-medium-4k-instruct'),
    ('mistralai/Mistral-7B-v0.3')
]

with open('continuity_scripts/data/counting_events_seq.json', 'r') as f:
    setups = json.load(f)

#setups = [
#    #('shop', 'Question: Alice goes to the shop. She buys a carton of milk. She buys an apple. She buys a potato. She buys a loaf of bread. How many items did Alice buy? Reply with a single-digit number\nAnswer: ', 'She', '.'),
#    #('zoo', 'Question: the class went to the zoo. They saw a lion. They saw an elephant. They saw a giraffe. They saw a penguin. How many animals did the class see? Reply with a single-digit number\nAnswer: ', 'They', '.'),
#    #('beach', 'Question: Emily went to the beach. She found a seashell. She found a starfish. She found a smooth stone. She found a piece of seaweed. How many things did Emily find? Reply with a single-digit number\nAnswer: ', 'She', '.')
#    ('zoo', 'Alice goes to the shop. She buys a carton of milk. She buys an apple. She buys a potato. She buys a loaf of bread.', 'How many items did Alice buy?'),
#]

for model_name in model_configs:
    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, data in list(enumerate(setups)):
        sentence = data['sentence']
        num_sentences = data['num_sentences']
        shrink_start = '-'
        shrink_end = '.'

        interesting_outputs = {}

        for j in range(0, 10):
            interesting_outputs[str(j)] = str(j)

        #interesting_outputs['Other Digits'] = [str(i) for i in range(10) if str(i) not in interesting_outputs]

        #sentence = '{passage}\nQuestion: {question} Reply with a single-digit number\nAnswer: '.format(passage=passage, question=question)
        model_name_short = model_name.lower().split("/")[1]
        save_path = f'results/batched/counting_events/{i}/{model_name_short}/counting_events-{i}-{model_name_short}'

        print(save_path)
        run_counting_experiment(current_model, current_tokenizer, sentence, None, shrink_start, shrink_end, interesting_outputs, save_path, 'Duration Factor', True, num_samples=40)
