import sys
sys.path.append('.')

import json

import numpy as np
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer

from continuity import get_outputs_from_continuous_inputs

from utils import get_tracked_tokens, add_results, plot_results



def run_counting_experiment(model, tokenizer, base_sentence, select_end, shrink_start, shrink_end, interesting_outputs, save_path, xlabel, legend=False, num_samples=100, select_start=None, raise_on_same_index=False, raise_on_different_index=False):
    if raise_on_different_index and raise_on_same_index:
        raise ValueError('raise_on_same_index and raise_on_different_index are mutually exclusive')
    
    tokenized = tokenizer(base_sentence, return_tensors='pt')
    detokenized = [tokenizer.decode(x, skip_special_tokens=True) for x in tokenized['input_ids'][0]]

    print(detokenized)

    start_position = 0
    max_position = len(detokenized)

    if select_start is not None:
        # Find the position of the first detokenized that contains select_start
        select_start_index = -1
        for i in range(len(detokenized)):
            if select_start in detokenized[i]:
                select_start_index = i
                break
        else:
            raise ValueError('select_start not found')

        start_position = select_start_index


    if select_end is not None:
        # Find the position of the first detokenized that contains select_end
        select_end_index = -1
        for i in range(len(detokenized)):
            if select_end in detokenized[i]:
                select_end_index = i
                break
        else:
            raise ValueError('select_end not found')

        assert select_end_index >= start_position
        max_position = select_end_index + 1
    

    # Find the position of the first detokenized that contains shrink_start
    if shrink_start is None:
        start_index = 0
    else:
        start_index = -1
        for i in range(start_position, max_position):
            if shrink_start in detokenized[i]:
                start_index = i
                break

    # Find the position of the last detokenized that contains shrink_end
    end_index = -1
    for i in range(max_position - 1, start_position - 1, -1):
        if shrink_end in detokenized[i]:
            end_index = i
            break

    if start_index == -1 or end_index == -1:
        raise ValueError('Shrink tokens not found')

    if raise_on_same_index and start_index == end_index:
        raise ValueError('Shrink tokens are the same')
    
    if raise_on_different_index and start_index != end_index:
        raise ValueError('Shrink tokens are different')

    print(start_index, end_index)
    print(detokenized[start_index:end_index + 1])

    # Loop through the vocabulary and find the tokens that contain the interesting outputs

    vocabulary_dict = tokenizer.get_vocab()
    tracked_tokens = get_tracked_tokens(vocabulary_dict, interesting_outputs)

    print(tracked_tokens)

    all_results = { x: [] for x in interesting_outputs }

    interpolation_factors = np.linspace(0.1, 1, num_samples, endpoint=True)

    for coefficient in interpolation_factors:
        position_ids = list(range(start_index))
        for i in range(start_index, end_index + 1):
            position_ids.append(start_index - 1 + (i - start_index + 1) * coefficient)

        for i in range(end_index + 1, len(detokenized)):
            position_ids.append(position_ids[-1] + 1)

        # print(list(zip(position_ids, detokenized)))

        position_ids = torch.tensor(position_ids) + 1

        # TODO: In teoria, nella forma base la posizione iniziale dovrebbe essere 0-indexed

        out = get_outputs_from_continuous_inputs(model, input_ids=tokenized['input_ids'], position_ids=position_ids, return_dict=True)
        #print(out['logits'].shape)

        probs = torch.softmax(out['logits'], dim=-1)
        print(tokenizer.decode([torch.argmax(probs[0, -1, :]).item()], skip_special_tokens=True), probs[0, -1, :].max().item())

        add_results(all_results, tracked_tokens, probs)

    plot_results(interpolation_factors, all_results, save_path, xlabel, legend=legend)
    with open(save_path + '.json', 'w') as f:
        json.dump(all_results, f)


# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b")
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
# model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
"""
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B") 

# model = AutoModelForCausalLM.from_pretrained(
#   "EleutherAI/pythia-6.9b-deduped",
#   revision="step143000",
#   cache_dir="./pythia-6.9b-deduped/step143000",
# )
# 
# tokenizer = AutoTokenizer.from_pretrained(
#   "EleutherAI/pythia-6.9b-deduped",
#   revision="step143000",
#   cache_dir="./pythia-6.9b-deduped/step143000",
# )

#tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
#model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

save_path = 'shrink-llama3-counting.png'

# base_sentence = (
#     "Q: Is a person who is 82 years old considered young or old? Reply with either \"young\" or \"old\"\n"
#     "A:"
# )
# shrink_start = '8'
# shrink_end = '2'
# print(list(range(len(detokenized))))
# base_sentence = 'The sum of 24 and 13 is '
# shrink_start = '1'
# shrink_end = '3'

# "apple apple" has 2 fruits
# "apple apple apple" has 3 fruits


base_sentence = 'Question: In the sentence "apple apple apple", how many times is a fruit mentioned? Reply with a single-digit number\nAnswer: '
#base_sentence = 'Question: In the word "apple", how many "p" letters are there? Reply with a single-digit number\nAnswer: '
shrink_start = 'apple'
shrink_end = 'apple'

interesting_outputs = {
    '1' : '1',
    '2' : '2',
    '3' : '3',
    'Other Digits' : ['0', '4', '5', '6', '7', '8', '9']
}
# interesting_outputs = {
#     'young' : 'you',
#     'old' : 'old'
# }

run_experiment(model, tokenizer, base_sentence, shrink_start, shrink_end, interesting_outputs, save_path)"""