import sys
sys.path.append('.')

import json

import numpy as np
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

from continuity import get_outputs_from_continuous_inputs
from utils import get_tracked_tokens, add_results, plot_results

from scipy.spatial import geometric_slerp

def pytorch_slerp(start, end, interpolation_factor):
    #print(start.shape, end.shape)
    # Convert to numpy
    start = start[0].detach().cpu().numpy()
    end = end[0].detach().cpu().numpy()

    results = np.zeros_like(start)


    for i in range(start.shape[0]):
      # Do the slerp
      if (start[i] == end[i]).all():
        results[i] = start[i]
      else:
        norm_start = np.linalg.norm(start[i], ord=2)
        norm_end = np.linalg.norm(end[i], ord=2)
        norm_result = (1 - interpolation_factor) * norm_start + interpolation_factor * norm_end

        normalized_start = start[i] / norm_start
        normalized_end = end[i] / norm_end

        #print(norm_start, norm_end, norm_result)
        print(np.linalg.norm(normalized_start, ord=2), np.linalg.norm(normalized_end, ord=2))

        results[i] = geometric_slerp(normalized_start, normalized_end, interpolation_factor) * norm_result

    # Convert back to PyTorch
    return torch.tensor(results).unsqueeze(0)

def run_embedding_interpolation_experiment(model, tokenizer, sentence_a, sentence_b, interesting_outputs, interpolation_technique, save_path, xlabel, legend, num_samples=100):
    tokenized_a = tokenizer(sentence_a, return_tensors='pt')
    tokenized_b = tokenizer(sentence_b, return_tensors='pt')

    embedding_function = model.get_input_embeddings()
    embedding_a = embedding_function(tokenized_a["input_ids"])
    embedding_b = embedding_function(tokenized_b["input_ids"])

    if interesting_outputs is None:
        all_results = []
    else:
        vocabulary_dict = tokenizer.get_vocab()
        tracked_tokens = get_tracked_tokens(vocabulary_dict, interesting_outputs)

        all_results = { x: [] for x in interesting_outputs }
    interpolation_factors = np.linspace(0, 1, num_samples, endpoint=True)

    for interpolation_factor in interpolation_factors:
        if interpolation_technique == 'linear':
            interpolated_embedding = (1 - interpolation_factor) * embedding_a + embedding_b * interpolation_factor
        elif interpolation_technique == 'spherical':
            interpolated_embedding = pytorch_slerp(embedding_a, embedding_b, interpolation_factor)
        else:
            raise ValueError('Invalid interpolation technique')

        # Compute the logits for the interpolated embeddings
        out = get_outputs_from_continuous_inputs(model, embeddings=interpolated_embedding, return_dict=True)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1)

        print(tokenizer.decode([torch.argmax(probs[0, -1, :]).item()], skip_special_tokens=True), probs[0, -1, :].max().item())
        # Print the top 5 tokens
        print(tokenizer.decode(torch.topk(probs[0, -1, :], 10).indices, skip_special_tokens=True))


        if interesting_outputs is None:
            all_results.append(probs[0, -1, :])
        else:
            add_results(all_results, tracked_tokens, probs)

    if interesting_outputs is None:
        interesting_tokens = set()

        for result in all_results:
            interesting_tokens.update([i for i in range(tokenizer.vocab_size) if result[i] > 0.05])
        
        print(interesting_tokens)

        interesting_tokens = [
            (x, tokenizer.decode([x], skip_special_tokens=True).strip())
            for x in interesting_tokens
        ]

        parsed_outputs = { name: [] for _, name in interesting_tokens }

        for result in all_results:
            for token_id, token_name in interesting_tokens:
                parsed_outputs[token_name].append(result[token_id].item())
        all_results = parsed_outputs
    
    plot_results(interpolation_factors, all_results, save_path, xlabel, legend=legend)

    with open(save_path + '.json', 'w') as f:
        json.dump(all_results, f)
