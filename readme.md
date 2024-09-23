# NLGift Benchmark

This is the official GitHub repository to our paper: [Can LLM Graph Reasoning Generalize beyond Pattern Memorization?](https://arxiv.org/abs/2406.15992). Our paper is accepted by EMNLP 2024 Findings.

Abstract: Large language models (LLMs) demonstrate great potential for problems with implicit graphical structures, while recent works seek to enhance the graph reasoning capabilities of LLMs through specialized instruction tuning. The resulting 'graph LLMs' are evaluated with in-distribution settings only, thus it remains underexplored whether LLMs are learning generalizable graph reasoning skills or merely memorizing patterns in the synthetic training data. To this end, we propose the NLGift benchmark, an evaluation suite of LLM graph reasoning generalization: whether LLMs could go beyond semantic, numeric, structural, reasoning patterns in the synthetic training data and improve utility on real-world graph-based tasks. Extensive experiments with two LLMs across four graph reasoning tasks demonstrate that while generalization on simple patterns (semantic, numeric) is somewhat satisfactory, LLMs struggle to generalize across reasoning and real-world patterns, casting doubt on the benefit of synthetic graph tuning for real-world tasks with underlying network structures. We explore three strategies to improve LLM graph reasoning generalization, and we find that while post-training alignment is most promising for real-world tasks, empowering LLM graph reasoning to go beyond pattern memorization remains an open research question.

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
