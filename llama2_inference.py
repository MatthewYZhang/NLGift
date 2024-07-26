from llama2_exp import LlamaBaseExp, LlamaFinetuneExp
import argparse
import warnings
warnings.filterwarnings('ignore')

# build some arguments, especially the training hyperparameters


def main():
    parser = argparse.ArgumentParser(description='arguments for model and training')
    # args for init experiment
    parser.add_argument('--base_model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')

    # args for inference
    parser.add_argument('--new_model_name', type=str, required=True)
    parser.add_argument('--base', action='store_true', help='If add this argument, will use base to do inference; else use finetune model')
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--encoding_method', type=str, required=True)
    parser.add_argument('--pattern', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--zeroshot_format', action='store_true', help='If add this argument, will use zeroshot_format')

    args = parser.parse_args()
    if args.base:
        print('using base model')
        exp = LlamaBaseExp(args)
        exp.evaluate(args)
    else:
        print('using finetuning model')
        exp = LlamaFinetuneExp(args)
        exp.evaluate(args)



if __name__ == '__main__':
    main()

