from llama2_exp import LlamaFinetuneExp, LlamaBaseExp
import argparse

# build some arguments, especially the training hyperparameters


def main():
    parser = argparse.ArgumentParser(description='arguments for model and training')
    # args for init experiment
    parser.add_argument('--base_model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    
    # args for trainer with default value
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # args for trainer
    parser.add_argument('--encoding_method', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--pattern', type=str) # which pattern we are interested in
    parser.add_argument('--train_data_path', type=str, required=True)


    args = parser.parse_args()
    
    args.new_model_name = args.train_data_path.replace('/', '__')
    exp = LlamaFinetuneExp(args)
    
    exp.finetune(args)
    # exp.evaluate()



if __name__ == '__main__':
    main()

