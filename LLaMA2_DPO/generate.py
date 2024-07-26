import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse
import json
import graph_description

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='arguments for DPO inference')

parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--zeroshot_format', action='store_true', help='If add this argument, will use zeroshot_format')
# parser.add_argument('--output_model', type=str, required=True)

args = parser.parse_args()

model_name = args.model_name
data_name = args.data_name
task = args.task
zeroshot_format = args.zeroshot_format

# model_name ="./merged_peft/final_merged_checkpoint"
adapter_path = f"{model_name}/final_checkpoint"
# adapter_path = "./dpo_results/final_checkpoint"

compute_dtype = getattr(torch, "float16")

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=compute_dtype,
    load_in_4bit=True,
)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     load_in_4bit=True,
# )

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

test_data = open(f'../LLM_Graph_Reasoning/{data_name}', 'r').readlines()


synthetic_task_list = ['connectivity', 'shortest_path', 'topology', 'flow']

encoding_method = 'incident'

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=4096)
        


with open(f'../LLM_Graph_Reasoning/experiment_logs/improvements/{task}/log/Llama-2_{model_name}_DPO_{data_name.split("/")[-1]}.jsonl', 'w') as file:
    for i, d in enumerate(test_data):
        item = json.loads(d)
                # print(i, item)
        limit = 500 if task in synthetic_task_list else 1000
        if i >= limit:
            break
        # print(encoding_method)
        if task == 'connectivity':
            problem = graph_description.ConnectivityProblem(encoding_method, item)
            prompt = problem.output_prompt
            answer = problem.output_answer
        elif task == 'shortest_path':
            problem = graph_description.ShortestPathProblem(encoding_method, item)
            prompt = problem.output_prompt
            answer = problem.output_answer
        elif task == 'topology':
            problem = graph_description.TopologyProblem(encoding_method, item)
            prompt = problem.output_prompt
            answer = problem.output_answer
        elif task == 'flow':
            problem = graph_description.FlowProblem(encoding_method, item)
            prompt = problem.output_prompt
            answer = problem.output_answer
        else:
            prompt = item['question']
            answer = item['answer']

        # print('output_prompt length: ', len(problem.output_prompt))
        # print(problem.output_prompt)
        
        if zeroshot_format:
            prompt = prompt.strip('\nA:') + graph_description.zeroshot_template_dict[task]
        try:
            
            result = pipe(f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Please answer strictly according to the question.\n<</SYS>>\n {prompt}\n[/INST]")
            response = result[0]['generated_text'].split('[/INST]')[-1]
            # print(response)
            # break
        except:
            print('WARNING: unexpected error, skipping example')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            response = 'None'
        if i % 50 == 0:
            print(i)
            # print(problem.output_prompt, problem.output_answer + '\n\n', f'response: {response}', end='\n\n')
        data_dict = {
            "encoded_question": prompt,
            "ground truth": answer,
            "response": response
        }
        data_dict = {**item, **data_dict}
        file.write(json.dumps(data_dict) + '\n')
    