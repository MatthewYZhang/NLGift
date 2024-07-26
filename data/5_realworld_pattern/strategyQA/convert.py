import json


zeroshot_prompt_template = 'Please answer the following question with either yes or no: {question}'
graph_prompt_template = 'Please try to build a relation graph to understand the following question and answer the question with either yes or no. Requirement: The last word of your response must be one of the following: Yes, No, Unknown. \nQuestion: {question}'



for output_format in ['graph', 'zeroshot']:
    output_file_path = f'{output_format}_test_data.jsonl'
    template = zeroshot_prompt_template if 'zeroshot' == output_format else graph_prompt_template
    with open(output_file_path, 'w') as f:
        items = open('test_data.jsonl', 'r').readlines()
        for item in items:
            it = json.loads(item)
            f.write(json.dumps({'question':template.format(question=it['question']), 'answer':it['answer']}) + '\n')

    
