import sys
sys.path.append('.')
import json
import pathlib

from transformers import AutoModelForCausalLM, AutoTokenizer
from continuity_tests.duration_shrink import run_counting_experiment


model_configs = [
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Llama-2-13b-chat-hf',
    'google/gemma-7b',
    'google/gemma-2-9b',
    'microsoft/Phi-3-medium-4k-instruct',
    'mistralai/Mistral-7B-v0.3',
]

with open('continuity_scripts/data/words_with_repetitions.json') as f:
    dataset = json.load(f)

for model_name in model_configs:

    current_model = AutoModelForCausalLM.from_pretrained(model_name)
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # All digits are interesting
    interesting_outputs = { str(i) : str(i) for i in range(10) }

    for i, element in enumerate(dataset):
        word = element['word']
        repetitions = element['repetitions']
        category_with_article = element['category_with_article']
        sentence = f'Question: In the sentence "{" ".join([word] * repetitions)}", how many times is {category_with_article} mentioned? Reply with a single-digit number\nAnswer: '
        #variant_sentence = f'Question: How many {category}s are listed in the sentence "{" ".join([word] * repetitions)}"? Reply with a single-digit number\nAnswer: '

        save_path = f'results/batched/counting/{i}/counting-{i}-{model_name.lower().split("/")[1]}'

        if pathlib.Path(save_path + '.json').exists():
            continue

        shrink_start = word
        shrink_end = word

        #used_sentence = sentence if prompt_type == 'standard' else variant_sentence

        try:
            run_counting_experiment(current_model, current_tokenizer, sentence, None, shrink_start, shrink_end, interesting_outputs, save_path, 'Duration Factor', legend=True)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error with {model_name} and {word}: {e}")

            try:
                with open('continuity_scripts/batched/counting_failed.json') as f:
                    failed = json.load(f)
            except:
                failed = []

            failed.append({
                'model': model_name,
                'word': word,
                'repetitions': repetitions,
                'category_with_article': category_with_article,
                'error': str(e)
            })

            pathlib.Path('continuity_scripts/batched').mkdir(parents=True, exist_ok=True)

            with open('continuity_scripts/batched/counting_failed.json', 'w') as f:
                json.dump(failed, f)