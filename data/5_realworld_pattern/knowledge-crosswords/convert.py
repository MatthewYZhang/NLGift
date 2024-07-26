import pandas as pd
import json
import random
import itertools

# Load the TSV file


zeroshot_prompt_template = 'Please select one option that satisfies all the constraints in the question. Please note that the {blank_num} words in each option are from blank 1 to {blank_num}. The question is: {question}.\nOptions:\n{option}\nYou should respond a single letter from A to D.'
graph_prompt_template = 'Please try to build a graph of the ralationship for the following question, and then select one option that satisfies all the constraints in this graph. Please note that the {blank_num} words in each option are from blank 1 to {blank_num}. The question is: {question}.\nOptions:\n{option}\nYour last sentence should give a single letter from A to D.'



# Function to convert a row into a JSON object with the specified format
def row_to_json(json_line):
    global graph_prompt_template, zeroshot_prompt_template

    prompt = graph_prompt_template if output_format == 'graph' else zeroshot_prompt_template

    data = json.loads(json_line)

    questions = [f"{src} {rel} {tgt}" for src, rel, tgt in zip(data["source"], data["relation"], data["target"])]
    result = {
        "questions": "",
        "options": "",
        "correct_answer": ""
    }
    blank_num = len(data["blanks"])
    all_options = list(itertools.product(*[data["options"][blank] for blank in data["blanks"]]))

    correct_answer = tuple(data["answer_all"])
    all_options.remove(correct_answer)
    wrong_answers = random.sample(all_options, 3)

    all_answers = [correct_answer] + wrong_answers
    random.shuffle(all_answers)

    options_list = []
    correct_option = ""

    for idx, answer in enumerate(all_answers):
        option_letter = chr(ord('A') + idx)
        option_string = f"{option_letter}. {', '.join(answer)}"
        options_list.append(option_string)
        if answer == correct_answer:
            correct_option = option_letter

    result["questions"] = ', '.join(questions)
    result["options"] = '\n'.join(options_list)
    result["answer"] = correct_option

    # print(result)
    question = prompt.format(blank_num=blank_num, question=result['questions'], option=result['options'])


    return json.dumps({'question': question, 'answer': correct_option}) + '\n'


file_path = 'test_data.jsonl'
data = open(file_path, 'r').readlines()

for output_format in ['graph', 'zeroshot']:
    random.seed(42)
    output_file_path = f'{output_format}_test_data.jsonl'
    with open(output_file_path, 'w') as f:
        
        for item in data:
            f.write(row_to_json(item))
    
    
