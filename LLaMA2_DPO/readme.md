# LLaMA2 DPO

The code for fine-tuning LLaMA2 using DPO is modified from [GitHub](https://github.com/mzbac/llama2-fine-tune/tree/master). DPO is used in our improvement section.

## Preparation

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U -r requirements.txt
```

If you see the error from the training script. `LlamaConverter requires the protobuf library but it was not found in your environment. `

Checkout the instructions on the installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.

```sh
pip install protobuf
```

## Fine-tuning with DPO

We give an example command to run DPO. Please specify model path `path/to/your/model`. Fine-tuned DPO model will be saved to current directory.

```bash
python dpo_trainer.py --model_name <path/to/your/model> --data_name ../data/6_improvements/alignment/gpt3.5t_align_conn-path-mix.jsonl
```

## Inference

We give an example command to do inference using DPO fine-tuned model. Please specify model path `path/to/your/model`.

```bash
python generate.py --task strategyQA --model_name <path/to/your/model> --data_name ../data/6_improvements/real-strategyqa_test_data.jsonl
```

