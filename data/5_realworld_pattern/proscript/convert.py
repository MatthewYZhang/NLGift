import pandas as pd
import json
import random

# Load the TSV file


zeroshot_prompt_template = 'Identify the logical order of all the following steps to achieve the following goal. Note that the numbering of the steps does not indicate their execution order, and your response should include all steps.\nGoal: {goal}\nSteps: {steps}\nFormat your response as a sequence, using "->" to separate (e.g., "step8->step4->step3").'
graph_prompt_template = 'Build a graph to understand the relationships and logic behind following steps and following goal, and then identify the logical order of all the following steps to achieve the following goal. Note that the numbering of the steps does not indicate their execution order, and your response should include all steps.\nGoal: {goal}\nSteps: {steps}\nOnce you understand the constraints, format your response as a sequence. For the sequence, format your response as a sequence, using "->" to separate (e.g., "step8->step4->step3").'



# Function to convert a row into a JSON object with the specified format
def row_to_json(item):
    global graph_prompt_template, zeroshot_prompt_template

    prompt = graph_prompt_template if output_format == 'graph' else zeroshot_prompt_template
    goal = item["scenario"]
    steps = item["flatten_input_for_edge_prediction"]
    question = prompt.format(goal=goal, steps=steps)
    answer = [tuple(i.split(' -> ')) for i in item["flatten_output_for_edge_prediction"].split('; ')]
    return json.dumps({"question": question, "answer": answer}) + '\n'


file_path = 'test.jsonl'
data = open(file_path, 'r').readlines()


for output_format in ['graph', 'zeroshot']:
    random.seed(42)
    ans = []

    for item in data:
        it = json.loads(item)
        ans.append(row_to_json(it))

    ans = random.sample(ans, 1000)

    output_file_path = f'{output_format}_test_data.jsonl'
    with open(output_file_path, 'w') as f:
        f.write(''.join(ans))
