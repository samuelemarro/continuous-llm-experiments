import sys
sys.path.append('.')

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.scaling import run_scaling_experiment

sentences = [
    ('france', 'The capital of France is'),
    ('gatsby', 'The Great Gatsby is my favourite'),
    ('romeo', 'O Romeo, Romeo, wherefore art thou')
]

model_names = [
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Llama-2-13b-chat-hf',
    'google/gemma-7b',
    'google/gemma-2-9b',
    'microsoft/Phi-3-medium-4k-instruct',
    'mistralai/Mistral-7B-v0.3'
]


for model_name in model_names:
    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for sentence_type, sentence in sentences:
        save_path = f'results/scaling/{sentence_type}/scaling-{model_name.lower().split("/")[1]}-{sentence_type}'
        run_scaling_experiment(current_model, current_tokenizer, sentence, 0.05, save_path, 'Scaling Factor', True)