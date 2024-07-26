import re
import json
import graph_description
import os
from openai import OpenAI
import sys

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_response(graph_question, model="gpt-3.5-turbo"):
    completion = client.chat.completions.create(
        model=model,
        max_tokens=4000,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please answer strictly according to the question."},
            {"role": "user", "content": graph_question}
        ]
    )
    return completion.choices[0].message.content

task = 'strategyQA'

log_directory = f'experiment_logs/realistic/{task}/'

def get_train_data_name(model_name):
    if 'conn-topo' in model_name:
        return 'conn-topo-mix'
    if 'full' in model_name:
        return 'full-mix'
    if 'conn-path' in model_name:
        return 'conn-path-mix'
    return 'gpt3.5t'
    

def evaluate_all(model_list : list[str], limit=1000, method='zeroshot'):
    for model in model_list:
        train_data_name = get_train_data_name(model)
        test_data_dir = f'realistic_data/{task}/{method}_test_data.jsonl'
        target_dir = f'{log_directory}gpt_{train_data_name}_{task}_{method}.jsonl'
        if os.path.exists(target_dir):
            save_file = open(target_dir, 'r')
            lines = save_file.readlines()
            if len(lines) == 1000:
                print(f'{target_dir} already has the data')
                continue
        with open(target_dir, 'w') as file:
            test_data = open(test_data_dir, 'r')
            for i, line in enumerate(test_data):
                item = json.loads(line)
                if i >= limit:
                    break
                response = get_response(item['question'], model=model)
                data_dict = {
                    "encoded_question": item['question'],
                    "ground truth": item['answer'],
                    "response": response
                }
                file.write(json.dumps(data_dict) + '\n')

limit = 1000

model_list = [
    'gpt-3.5-turbo',
    # Your fine-tuned model here
]

evaluate_all(model_list, limit, method='zeroshot')
print('zeroshot strategyQA finished')

