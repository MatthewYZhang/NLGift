import json
import random

# Load the JSONL file
file_path = 'strategyqa_train.json'

# Read the JSONL file with error handling for JSONDecodeError
with open(file_path, 'r') as file:
    data = json.load(file)
# print(data, type(data))
# Set the random seed
random.seed(42)

# Randomly select 500 samples
selected_samples = random.sample(data, 1000)

# Convert the selected samples back to JSONL format
selected_jsonl = '\n'.join([json.dumps(sample) for sample in selected_samples])

# Save the selected samples to a new file
output_file_path = 'test_data.jsonl'
with open(output_file_path, 'w') as file:
    file.write(selected_jsonl)

output_file_path
