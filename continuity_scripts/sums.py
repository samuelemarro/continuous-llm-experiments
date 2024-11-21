import sys
sys.path.append('.')

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.duration_shrink import run_counting_experiment


model_configs = [
    ('meta-llama/Meta-Llama-3-8B', 'variant1'),
    ('meta-llama/Llama-2-13b-chat-hf', 'standard'),
    ('google/gemma-7b', 'standard'),
    ('google/gemma-2-9b', 'standard'),
    ('microsoft/Phi-3-medium-4k-instruct', 'standard'),
    ('mistralai/Mistral-7B-v0.3', 'standard')
]

setups = [
    (24, 13),
    (32, 56),
    (13, 74)
]

current_model = None
current_tokenizer = None

for model_name, prompt_type in model_configs:

    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for number_1, number_2 in setups:
        #sentence = f'Question: In the sentence "{" ".join([word] * repetitions)}", how many times is {category_with_article} mentioned? Reply with a single-digit number\nAnswer: '
        #variant_sentence = f'Question: How many {category}s are listed in the sentence "{" ".join([word] * repetitions)}"? Reply with a single-digit number\nAnswer: '

        sentence = f'The sum of {number_1} and {number_2} is '
        sentence_variant = f'The sum of {" ".join(str(number_1))} and {" ".join(str(number_2))} is '

        first_digit_number_2 = int(str(number_2)[0])
        second_digit_number_2 = int(str(number_2)[1])

        interesting_digits = [
            str(number_1 + number_2)[0],
            str(number_1 + first_digit_number_2)[0],
            str(number_1 + second_digit_number_2)[0]
        ]
        interesting_outputs = {i : i for i in interesting_digits}

        save_path = f'results/shrink/sums/{number_1}_{number_2}/shrink-{model_name.lower().split("/")[1]}-{number_1}_{number_2}-sums'

        shrink_start = str(first_digit_number_2)
        shrink_end = str(second_digit_number_2)

        if prompt_type == 'standard':
            run_counting_experiment(current_model, current_tokenizer, sentence, None, shrink_start, shrink_end, interesting_outputs, save_path, 'Duration Factor', True)
        elif prompt_type == 'variant1':
            run_counting_experiment(current_model, current_tokenizer, sentence_variant, None, shrink_start, shrink_end, interesting_outputs, save_path, 'Duration Factor', True)
        else:
            raise ValueError('Invalid prompt type')
    


    
    #run_experiment(current_model, current_tokenizer, sentence, shrink_start, shrink_end, interesting_outputs, save_path)