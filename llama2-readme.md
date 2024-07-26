# Fine tuning Llama2

## Preparation

```bash
# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
export RUSTFLAGS="-A invalid_reference_casting"

# install python requirements
conda create -n llama2 python=3.12.2
conda activate llama2
pip install -r requirements.txt
```


## Fine-tuning

For fine-tuning, we give an example command to fine-tune on real-world task. Fine-tuned model will be saved to current directory.

```bash
# fine-tune
python llama2_finetune.py --pattern realistic --task real-full-mix  --encoding_method incident --train_data_path data/5_realworld_pattern/full-mix_train_data
```


## Inference

For evaluation, we give an example command to do inference on real-world task. Make sure you have the fine-tuned model ready at `path/to/your/model`. Inference results will be saved to `experiment_logs/<pattern>/<task>/log/`.

```bash
# inference using base model (LLaMA2-7B)
python llama2_inference.py --pattern realistic --new_model_name none --base --task strategyQA --encoding_method incident --test_data_path data/5_realworld_pattern/strategyQA/zeroshot_test_data

# inference using fine-tuned model
python llama2_inference.py --pattern realistic --new_model_name <path/to/your/model> --task strategyQA --encoding_method incident --test_data_path data/5_realworld_pattern/strategyQA/zeroshot_test_data
```