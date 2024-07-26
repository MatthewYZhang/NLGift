import random

# Set the random seed for reproducibility
random.seed(42)

# Define the filenames and the number of samples to select
filenames = {
    'MC_easy.jsonl': 334,  # 166 + 1 for the easy file
    'MC_medi.jsonl': 333,
    'MC_hard.jsonl': 333
}

selected_samples = []

# Function to select random lines from a file
def select_random_lines(filename, num_lines):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        selected = random.sample(lines, num_lines)
    return selected

# Collecting samples from each file
for filename, num_samples in filenames.items():
    selected_samples.extend(select_random_lines(filename, num_samples))

# Write the selected samples to the new file
with open('test_data.jsonl', 'w', encoding='utf-8') as outfile:
    for sample in selected_samples:
        outfile.write(sample)
