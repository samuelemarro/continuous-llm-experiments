import sys
sys.path.append('.')

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.embedding_interpolation import run_embedding_interpolation_experiment

repetitions = 4


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
    #('capital', 'The capital of [] is a city named', ('France', 'Paris', 'par'), ('Germany', 'Berlin', 'ber')),
    #('apple', 'The most common colour for the skin of an [] is', ('apple', 'Red', 'red'), ('avocado', 'Green', 'green'))
    #('colour', 'The most common colour of a [] is', ('banana', 'Yellow', 'yel'), ('cherry', 'Red', 'red'), [
    #    ('Orange', 'ora'),
    #    ('Other Colours', ['whi', 'bla', 'gre', 'blu', 'pur', 'bro', 'pin', 'gra'])
    #])
    #('sum', 'The sum of 2 and [] is ', ('3', ''))



    #('fruit_space', 'Question: Are [] fruits? (yes/no)\nAnswer:', ('apples', 'Yes', 'yes'), ('bananas', 'No', 'no'), [])
    #('fruit_red', 'Question: Are [] red? (yes/no)\nAnswer:', ('apples', 'Yes', 'yes'), ('bananas', 'No', 'no'), [])
    
    #('cats_dogs', 'animal_space', 'Question: Are [] animals? (yes/no)\nAnswer:', ('cats', 'Yes', 'yes'), ('dogs', 'No', 'no'), []),
    #('cats_dogs', 'animal_meow', 'Question: Do [] meow? (yes/no)\nAnswer:', ('cats', 'Yes', 'yes'), ('dogs', 'No', 'no'), []),
    
    #('forks_spoons', 'cutlery_space', 'Question: Are [] a type of utensil? (yes/no)\nAnswer:', ('forks', 'Yes', 'yes'), ('spoons', 'No', 'no'), []),
    #('forks_spoons', 'cutlery_soup', 'Question: Are [] used for soup? (yes/no)\nAnswer:', ('forks', 'Yes', 'yes'), ('spoons', 'No', 'no'), [])

    ('water_juice', 'drink_space', 'Question: Is [] a drink? (yes/no)\nAnswer:', ('water', 'Yes', 'yes'), ('juice', 'No', 'no'), []),
    ('water_juice', 'drink_sugar', 'Question: Does [] contain sugar? (yes/no)\nAnswer:', ('water', 'Yes', 'yes'), ('juice', 'No', 'no'), [])

    #('happy', '[] is a positive or negative emotion?')
]

first_element = True

interpolation_technique = 'linear'

for model_name, prompt_type in model_configs:

    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for experiment_category, name, sentence, (word_a, target_a, prefix_a), (word_b, target_b, prefix_b), extra_targets in setups:
        #if prompt_type == 'standard':
        #sentence_a = f'Question: In the sentence "{" ".join([word] * repetitions)}", how many times is {category_with_article} mentioned? Reply with a single-digit number\nAnswer: '
        #sentence_b = f'Question: In the sentence "{" ".join([word] + [other_word] * (repetitions - 1))}", how many times is {category_with_article} mentioned? Reply with a single-digit number\nAnswer: '
        #elif prompt_type == 'variant1':
        #sentence_a = f'Question: How many {category}s are listed in the sentence "{" ".join([word] * repetitions)}"? Reply with a single-digit number\nAnswer: '
        #sentence_b = f'Question: How many {category}s are listed in the sentence "{" ".join([word] + [other_word] * (repetitions - 1))}"? Reply with a single-digit number\nAnswer: '

        #sentence_a = f'Question: Consider the following list: "{" ".join([word] * repetitions)}". How many times is {category_with_article} mentioned? Reply with a single-digit number, including 0 if there are none.\nAnswer: '
        #sentence_b = f'Question: Consider the following list: "{" ".join([other_word] * repetitions)}". How many times is {category_with_article} mentioned? Reply with a single-digit number, including 0 if there are none.\nAnswer: '

        sentence_a = sentence.replace('[]', word_a)
        sentence_b = sentence.replace('[]', word_b)

        print(sentence_a)
        print(sentence_b)

        interesting_outputs = {
            target_a: [prefix_a],
            target_b: [prefix_b]
        }
        for extra_target, extra_prefixes in extra_targets:
            interesting_outputs[extra_target] = extra_prefixes
        #assert False

        save_path = f'results/embedding_interpolation/{experiment_category}/{name}/{interpolation_technique}/embedding_interpolation-{model_name.lower().split("/")[1]}-{name}-{experiment_category}-{interpolation_technique}'

        try:
            run_embedding_interpolation_experiment(current_model, current_tokenizer, sentence_a, sentence_b, interesting_outputs, interpolation_technique, save_path, 'Interpolation Factor', legend=True)
        except Exception as e:
            print(e)
            continue
        first_element = False
    #run_experiment(current_model, current_tokenizer, sentence, shrink_start, shrink_end, interesting_outputs, save_path)