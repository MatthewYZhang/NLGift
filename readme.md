# NLGift Benchmark

We provide the complete dataset for each pattern, the code to train and evaluate using both ChatGPT and LLaMA2-7B, and the code to DPO on LLaMA2-7B.

## Complete Dataset

All data used for training and evaluation are in the directory `data/`. 

## Fine-tuning ChatGPT and Evaluation

For fine-tuning ChatGPT, we directly use the GUI provided by [OpenAI](https://platform.openai.com/finetune/). For inference, please refer to `gpt_inference_example.py`.


## Fine-tuning LLaMA2-7B and Evaluation

Please refer to `llama2-readme.md` for step-to-step instructions for fine-tuning and evaluation.

## LLaMA2-7B DPO

Due to package updates, we choose to use another version of `transformers`. Please check `LLaMA2_DPO/` for details.

## Experiment results

You can find all of the inference results at `experiment_logs/`. 

Since we are doing inference on both in-distribution and out-of-distribution data, some responses may not be easily determined whether they are correct or not. Human labelling, keywords filtering, regular expressions are used to determine the correctness of the responses.