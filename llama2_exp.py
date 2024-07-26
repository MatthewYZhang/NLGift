# Import necessary libraries
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import gc
import json
import graph_description

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class LlamaBaseExp:
    def __init__(self, args) -> None:
        self.base_model_name = args.base_model_name
        # self.new_model = args.new_model
        self.new_model_name = self.base_model_name.split('/')[-1]
        self._config_quant()
        self._config_model()
        self._display_cuda_memory()

    

    def _config_quant(self):
        compute_dtype = getattr(torch, "float16")
        self.quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)
        print('quant config complete')

    def _config_model(self):
        # Load model with 4-bit precision
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, quantization_config=self.quant_config, device_map={"": 0})
        self.base_model.config.use_cache = False
        self.base_model.config.pretraining_tp = 1
        self.model = self.base_model
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        print('model config complete')
        
    def _display_cuda_memory(self):    
        print("\n--------------------------------------------------\n")
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print("\n--------------------------------------------------\n")

    def _load_data(self, args):
        self.task = args.task
        self.encoding_method = args.encoding_method
        data_path = args.train_data_path + '_llama.jsonl'
        dataset = load_dataset('json', data_files=data_path, split='train')
        return dataset
    
    def _generate_one(self, prompt):
        pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, max_length=len(prompt)+1000)
        result = pipe(f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Please answer strictly according to the question.\n<</SYS>>\n {prompt}\n[/INST]")
        response = result[0]['generated_text'].split('[/INST]')[-1]
        # print(response)
        return response
    
    def finetune(self, args):
        raise NotImplementedError
    
    def dpo(self, args):
        raise NotImplementedError
    
    def evaluate(self, args):
        encoding_method = args.encoding_method
        task = args.task
        pattern = args.pattern
        zeroshot_format = args.zeroshot_format
        test_data = open(args.test_data_path+'.jsonl', 'r')
        test_data_name = '-'.join(args.test_data_path.split('/')[-2:])
        synthetic_task_list = ['connectivity', 'shortest_path', 'topology', 'flow']
        with open(f'experiment_logs/{pattern}/{task}/log/Llama-2_{'_'.join(self.new_model_name.split('__')[-2:])}__{test_data_name}.jsonl', 'w') as file:
            for i, line in enumerate(test_data):
                item = json.loads(line) 
                # print(i, item)
                limit = 500 if task in synthetic_task_list else 1000
                if i >= limit:
                # TODO
                # if i > 0:
                    break
                # print(encoding_method)
                number_convert = float if 'float' in test_data_name else int
                if task == 'connectivity':
                    problem = graph_description.ConnectivityProblem(encoding_method, item)
                    prompt = problem.output_prompt
                    answer = problem.output_answer
                elif task == 'shortest_path':
                    problem = graph_description.ShortestPathProblem(encoding_method, item, number_convert=number_convert)
                    prompt = problem.output_prompt
                    answer = problem.output_answer
                elif task == 'topology':
                    problem = graph_description.TopologyProblem(encoding_method, item)
                    prompt = problem.output_prompt
                    answer = problem.output_answer
                elif task == 'flow':
                    problem = graph_description.FlowProblem(encoding_method, item, number_convert=number_convert)
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
                    response = self._generate_one(prompt)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('WARNING: ran out of memory, skipping example')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
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



class LlamaFinetuneExp(LlamaBaseExp):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.new_model_name = args.new_model_name

    def finetune(self, args):
        # self.data_name = data_name
        self.dataset = self._load_data(args)

        # Set PEFT Parameters
        self.peft_params = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM")
        # Define training parameters
        training_params = TrainingArguments(output_dir=f"./results/{args.task}/{args.new_model_name}", num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=1, optim="paged_adamw_32bit", save_steps=100, logging_steps=25, learning_rate=args.lr, weight_decay=0.001, fp16=False, bf16=False, max_grad_norm=0.3, max_steps=-1, warmup_ratio=0.03, group_by_length=True, lr_scheduler_type="constant", report_to="tensorboard")
        trainer = SFTTrainer(model=self.base_model, train_dataset=self.dataset, peft_config=self.peft_params, dataset_text_field="text", max_seq_length=None, tokenizer=self.tokenizer, args=training_params, packing=False)

        gc.collect()
        torch.cuda.empty_cache()

        trainer.train()
        trainer.model.save_pretrained(self.new_model_name)
        trainer.tokenizer.save_pretrained(self.new_model_name)

    def evaluate(self, args):
        # self.model = PeftModel.from_pretrained(self.model, self.new_model_name)
        # prompt = "Who is Donald J Trump?"
        # pipe = pipeline(task="text-generation", model=self.base_model, tokenizer=self.tokenizer, max_length=2000)
        # result = pipe(f"<s>[INST] {prompt} [/INST]")
        # print(result[0]['generated_text'])

        self.model = PeftModel.from_pretrained(self.base_model, self.new_model_name)
        super().evaluate(args)


