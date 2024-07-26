import json
import random

llama_format = '<s>[INST] <<SYS>>You are a helpful assistant.<</SYS>>{question}\nA:\n[/INST]{answer}</s>'

# Load the JSONL file
file_path = 'code_alpaca_2k.json'

# Read the JSONL file with error handling for JSONDecodeError
with open(file_path, 'r') as file:
    data = json.load(file)
# print(data, type(data))
# Set the random seed
random.seed(42)

# Randomly select 500 samples
selected_samples = random.sample(data, 200)

# Convert the selected samples back to JSONL format
selected_jsonl = '\n'.join([json.dumps({'text': llama_format.format(question=f'{sample['instruction']}\n{sample['input']}', answer=sample['output'])}) for sample in selected_samples]) + '\n'

# Save the selected samples to a new file
output_file_path = 'generalcode.jsonl'
with open(output_file_path, 'w') as file:
    file.write(selected_jsonl)

output_file_path
