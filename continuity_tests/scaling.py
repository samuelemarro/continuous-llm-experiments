import sys
sys.path.append('.')

import numpy as np
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer

from continuity import get_outputs_from_continuous_inputs

from utils import get_tracked_tokens, add_results, plot_results



def run_scaling_experiment(model, tokenizer, sentence, interest_threshold, save_path, xlabel, legend):
    tokenized = tokenizer(sentence, return_tensors='pt')
    tokens = tokenized['input_ids'][0]
    detokenized = [tokenizer.decode(x, skip_special_tokens=True) for x in tokens]
    print(detokenized)

    all_results = []

    interpolation_factors = np.linspace(0.01, 1.5, 100, endpoint=True)

    for interpolation_factor in interpolation_factors:
        print('Original input ids:', tokens.shape)
        
        position_ids = (torch.arange(len(tokens), device='cuda') + 1) * interpolation_factor
        position_ids = position_ids.to(tokenized['input_ids'].device)
        # TODO: Check if this is correct
        
        print('Position ids:', position_ids)
        

        out = get_outputs_from_continuous_inputs(model, input_ids=tokens.unsqueeze(0), position_ids=position_ids, return_dict=True)
        probs = torch.softmax(out['logits'], dim=-1)

        print(tokenizer.decode([torch.argmax(probs[0, -1, :]).item()], skip_special_tokens=True), probs[0, -1, :].max().item())

        all_results.append(probs[0, -1, :])
    
    interesting_tokens = set()

    for result in all_results:
        interesting_tokens.update([i for i in range(tokenizer.vocab_size) if result[i] > interest_threshold])
    
    print(interesting_tokens)

    interesting_tokens = [
        (x, tokenizer.decode([x], skip_special_tokens=True).strip())
        for x in interesting_tokens
    ]

    parsed_outputs = { name: [] for _, name in interesting_tokens }

    for result in all_results:
        for token_id, token_name in interesting_tokens:
            parsed_outputs[token_name].append(result[token_id].item())

    plot_results(interpolation_factors, parsed_outputs, save_path, xlabel, legend)
