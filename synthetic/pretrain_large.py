import os
import argparse
from argparse import Namespace
import yaml
import networkx as nx
from datasets import load_dataset
from data import create_tokenized_dataset, create_data_collator
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer

import torch.utils.tensorboard
import pdb


def get_args():
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument("--graph_dir", type=str, default="./graphs")
    parser.add_argument("--graph", type=str)
    parser.add_argument("--data_dir", type=str, default="./data_large")
    parser.add_argument("--trial", type=int)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--block_size", type=int, default=-1)
    
    # Model
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--hf_model_dir", type=str, default="./hf_models")
    parser.add_argument("--init", type=str, default="scratch")
    
    # Trainer's Hyperparameters
    parser.add_argument("--output_root_dir", type=str, default="./results_large")
    parser.add_argument("--logging_root_dir", type=str, default="./logging_large")
    parser.add_argument("--training_arguments_dir", type=str, default="./training_arguments")
    parser.add_argument("--training_arguments", type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    # Load training arguments
    training_arguments_path = os.path.join(args.training_arguments_dir, f"{args.training_arguments}.yaml")
    with open(training_arguments_path, "r") as f:
        args_training_arguments = yaml.safe_load(f)
    args = Namespace(**vars(args), **args_training_arguments)
    
    # Load dataset
    args.data_path = os.path.join(args.data_dir, f"{args.graph}_num_samples={args.num_samples}", f"samples_{args.trial}.pt")
    dataset = create_tokenized_dataset(args.data_path)
    
    # Load model
    model_path = os.path.join(args.hf_model_dir, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token = tokenizer.sep_token
    pretrained_model = AutoModelForMaskedLM.from_pretrained(model_path)
    if args.init == "pretrain":
        model = pretrained_model
    else:
        model = AutoModelForMaskedLM.from_config(pretrained_model.config)
    
    args.graph_path = os.path.join(args.graph_dir, args.graph)
    G = nx.read_edgelist(args.graph_path)
    START_TOKEN_ID = tokenizer.convert_tokens_to_ids('a')
    target_tokens = [tokenizer.convert_ids_to_tokens(START_TOKEN_ID + i) for i in range(G.number_of_nodes())]
    data_collator = create_data_collator(tokenizer, masking_method="random_target_token", target_tokens=target_tokens)

    if not os.path.exists(args.output_root_dir):
        os.mkdir(args.output_root_dir)

    if not os.path.exists(args.logging_root_dir):
        os.mkdir(args.logging_root_dir)

    args.output_dir = os.path.join(
        args.output_root_dir,
        f"{args.model}_{args.init}_graph={args.graph}_training_arguments={args.training_arguments}_trial_{args.trial}"
    )

    args.logging_dir = os.path.join(
        args.logging_root_dir,
        f"{args.model}_{args.init}_graph={args.graph}_training_arguments={args.training_arguments}_trial_{args.trial}"
    )
    
    # Train
    print("Training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=(args.save_total_limit if args.save_total_limit > 0 else None),
        logging_dir=args.logging_dir,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=data_collator,
    )

    trainer.train()