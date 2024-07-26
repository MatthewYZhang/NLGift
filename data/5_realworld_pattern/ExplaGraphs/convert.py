import pandas as pd
import json

zeroshot_prompt_template = 'Please judge if the following two sentences support each other or counter each other: "{}" and "{}". Respond in a single word, either "support" or "counter": '
graph_prompt_template = 'Please try to build a graph of the ralationship for the following two sentences, and judge if they support each other or counter each other: "{}" and "{}". The last sentence of your response must state clearly your judgement.'



# Function to convert a row into a JSON object with the specified format
def row_to_json(row):
    global graph_prompt_template, zeroshot_prompt_template

    prompt = graph_prompt_template if output_format == 'graph' else zeroshot_prompt_template
    question = prompt.format(row['sentence1'], row['sentence2'])
    answer = "support" if row['label'] == 'support' else "counter"
    return json.dumps({"question": question, "answer": answer}) + '\n'




for output_format in ['graph', 'zeroshot']:
    file_path = 'train.tsv'
    data = pd.read_csv(file_path, sep='\t', header=None, names=['sentence1', 'sentence2', 'label', 'context'])
    data = data.sample(1000, random_state=42)
    ans = ''
    for index, row in data.iterrows():
        ans += row_to_json(row)
    output_file_path = f'{output_format}_test_data.jsonl'
    with open(output_file_path, 'w') as f:
        f.write(ans)
