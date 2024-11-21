import sys
sys.path.append('.')

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.embedding_interpolation import run_embedding_interpolation_experiment

#interesting_outputs = {}
#
#for i in range(0, repetitions + 1):
#    interesting_outputs[str(i)] = str(i)
#
#interesting_outputs['Other Digits'] = [str(i) for i in range(10) if str(i) not in interesting_outputs]

#print(interesting_outputs)

model_configs = [
    ('meta-llama/Meta-Llama-3-8B', 'standard'),
    ('meta-llama/Llama-2-13b-chat-hf', 'variant1'),
    ('google/gemma-7b', 'variant1'),
    ('google/gemma-2-9b', 'standard'),
    ('microsoft/Phi-3-medium-4k-instruct', 'standard'),
    ('mistralai/Mistral-7B-v0.3', 'variant1')
]

setups = [
    ('capital', 'The capital of [] is a city named', ('France', 'Paris', 'par'), ('Germany', 'Berlin', 'ber')),
    ('apple', 'The most common colour for the skin of an [] is', ('apple', 'Red', 'red'), ('avocado', 'Green', 'green'))
    ('colour', 'The most common colour of a [] is', ('banana', 'Yellow', 'yel'), ('cherry', 'Red', 'red'), [
        ('Orange', 'ora'),
        ('Other Colours', ['whi', 'bla', 'gre', 'blu', 'pur', 'bro', 'pin', 'gra'])
    ])


    ('fruit_usage', 'Alice bought some [] at the', 'apples', 'bananas'),
    ('fruit_color', 'The most common colour of [] is', 'apples', 'bananas')

    ('cats_dogs', 'animal_usage', 'We bought two [] at the', 'cats', 'dogs'),
    ('cats_dogs', 'animal_kept', '[] are usually kept as', 'cats', 'dogs'),
    ('cats_dogs', 'animal_repeat', 'Question: Repeat the word []. Answer:', 'cats', 'dogs'),
    
    ('forks_spoons', 'cutlery_usage', 'We used the [] to eat', 'forks', 'spoons'),
    ('forks_spoons', 'cutlery_material', 'Typically, [] are made of', 'forks', 'spoons'),
    ('forks_spoons', 'cutlery_repeat', 'Question: Repeat the word []. Answer:', 'forks', 'spoons'),

    ('water_juice', 'drink_usage', 'We drank some [] in the', 'water', 'juice'),
    ('water_juice', 'drink_repeat', 'Question: Repeat the word []. Answer:', 'water', 'juice'),


    #('fruit_name', 'Question: Repeat the word []. Answer:', 'apples', 'bananas')
    #('happy', '[] is a positive or negative emotion?')
]

first_element = True

interpolation_technique = 'linear'

for model_name, prompt_type in model_configs:

    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for experiment_name, name, sentence, word_a, word_b in setups:
        sentence_a = sentence.replace('[]', word_a)
        sentence_b = sentence.replace('[]', word_b)

        print(sentence_a)
        print(sentence_b)


        save_path = f'results/embedding_interpolation_usage/{experiment_name}/{name}/{interpolation_technique}/embedding_interpolation-{model_name.lower().split("/")[1]}-{experiment_name}-{name}-{interpolation_technique}'

        try:
            run_embedding_interpolation_experiment(current_model, current_tokenizer, sentence_a, sentence_b, None, interpolation_technique, save_path, 'Interpolation Factor', legend=True)
        except Exception as e:
            print(e)
            pass