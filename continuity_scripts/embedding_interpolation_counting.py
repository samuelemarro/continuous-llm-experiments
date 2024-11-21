import sys
sys.path.append('.')

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.embedding_interpolation import run_embedding_interpolation_experiment

repetitions = 4


interesting_outputs = {}

for i in range(0, repetitions + 1):
    interesting_outputs[str(i)] = str(i)

interesting_outputs['Other Digits'] = [str(i) for i in range(10) if str(i) not in interesting_outputs]

print(interesting_outputs)

model_configs = [
    ('meta-llama/Meta-Llama-3-8B', 'standard'),
    ('meta-llama/Llama-2-13b-chat-hf', 'variant1'),
    ('google/gemma-7b', 'variant1'),
    ('google/gemma-2-9b', 'standard'),
    ('microsoft/Phi-3-medium-4k-instruct', 'standard'),
    ('mistralai/Mistral-7B-v0.3', 'variant1')
]

setups = [
    ('apple', 'cash', 'fruit', 'a fruit'),
    ('cat', 'rock', 'animal', 'an animal'),
    ('rose', 'plane', 'flower', 'a flower')
]

first_element = True

interpolation_technique = 'spherical'

for model_name, prompt_type in model_configs:

    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for word, other_word, category, category_with_article in setups:
        #if prompt_type == 'standard':
        #sentence_a = f'Question: In the sentence "{" ".join([word] * repetitions)}", how many times is {category_with_article} mentioned? Reply with a single-digit number\nAnswer: '
        #sentence_b = f'Question: In the sentence "{" ".join([word] + [other_word] * (repetitions - 1))}", how many times is {category_with_article} mentioned? Reply with a single-digit number\nAnswer: '
        #elif prompt_type == 'variant1':
        #sentence_a = f'Question: How many {category}s are listed in the sentence "{" ".join([word] * repetitions)}"? Reply with a single-digit number\nAnswer: '
        #sentence_b = f'Question: How many {category}s are listed in the sentence "{" ".join([word] + [other_word] * (repetitions - 1))}"? Reply with a single-digit number\nAnswer: '

        #sentence_a = f'Question: Consider the following list: "{" ".join([word] * repetitions)}". How many times is {category_with_article} mentioned? Reply with a single-digit number, including 0 if there are none.\nAnswer: '
        #sentence_b = f'Question: Consider the following list: "{" ".join([other_word] * repetitions)}". How many times is {category_with_article} mentioned? Reply with a single-digit number, including 0 if there are none.\nAnswer: '

        #sentence_a = 'Question: Alice goes to the shop. She buys an apple. She buys a banana. She buys an apple. She buys an apple. How many apples did Alice buy? Reply with a single-digit number\nAnswer: '
        #sentence_b = 'Question: Alice goes to the shop. She buys a pear. She buys a banana. She buys a pear. She buys a pear. How many apples did Alice buy? Reply with a single-digit number\nAnswer: '

        sentence_a = 'Question: How many apples appear in the sentence "apple apple apple apple"? Reply with a single-digit number, including 0 if there are none.\nAnswer: '
        sentence_b = 'Question: How many apples appear in the sentence "pear pear pear pear"? Reply with a single-digit number, including 0 if there are none.\nAnswer: '

        print(sentence_a)
        print(sentence_b)
        #assert False

        save_path = f'results/embedding_interpolation_counting/counting/{word}/{interpolation_technique}/embedding_interpolation-{model_name.lower().split("/")[1]}-{word}-counting-{interpolation_technique}'


        run_embedding_interpolation_experiment(current_model, current_tokenizer, sentence_a, sentence_b, interesting_outputs, interpolation_technique, save_path, 'Interpolation Factor', legend=True)
        first_element = False
    #run_experiment(current_model, current_tokenizer, sentence, shrink_start, shrink_end, interesting_outputs, save_path)