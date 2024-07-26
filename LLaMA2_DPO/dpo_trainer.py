import os
import torch

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from utils import find_all_linear_names, print_trainable_parameters
import argparse

parser = argparse.ArgumentParser(description='arguments for DPO')

parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--data_name', type=str, required=True)
# parser.add_argument('--output_model', type=str, required=True)

args = parser.parse_args()




# model_name = "../LLM_Graph_Reasoning/old_models/reasoning_data__connectivity__model"
# data_name = "improvements/alignment/gpt3.5t_align_connectivity_part.jsonl"

model_name = args.model_name
data_name = args.data_name
output_dir = f"{model_name.split('/')[-1]}_DPO_{data_name.split('/')[-1][:-6]}"

print(f"EXPERIMENT START!!! model name is {model_name}, and output path is {output_dir}")


dataset = load_dataset("json", data_files=data_name, split="train")

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"": 0})
model.config.use_cache = False
# model.load_adapter(model_name, adapter_name='reference')
model = prepare_model_for_kbit_training(model)

model_ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=compute_dtype, quantization_config=bnb_config, device_map={"": 0})

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

training_args = DPOConfig(
    beta=0.1,
    output_dir='./results',
    per_device_train_batch_size=1,
)

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=find_all_linear_names(model),
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

# dpo_trainer = DPOTrainer(
#     model,
#     args=training_args,
#     train_dataset=dataset,
#     tokenizer=tokenizer,
#     model_adapter_name="train",
#     ref_adapter_name="reference",
#     max_prompt_length=512,
#     max_length=1024,
# )

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_prompt_length=512,
    max_length=768,
)


dpo_trainer.train()
dpo_trainer.save_model(output_dir)


output_dir = os.path.join(output_dir, "final_checkpoint")
dpo_trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"EXPERIMENT DONE!!! model name is {model_name}, and output path is {output_dir}")