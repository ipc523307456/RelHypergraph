import os
import argparse
import random
import string
import networkx as nx
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="./graphs")
    parser.add_argument("--graph", type=str)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--data_root_dir", type=str, default="./data")
    
    # Model
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--hf_model_dir", type=str, default="./hf_models")
    
    return parser.parse_args()
            
def generate_samples_from_graph(G, vocab, num_samples):        
    edges = []
    weights = []
    
    for u, v, d in G.edges(data=True):
        edges.append([u, v])
        weights.append(d['weight'])
    sample_edges = random.choices(edges, weights, k=num_samples)
    sample_strs = [generate_sample_str(edge, vocab) for edge in sample_edges]
    
    return sample_strs

def generate_sample_str(edge, vocab):
    shuffled_edge = random.sample(edge, len(edge))
    sample_str = " ".join([vocab[int(node)] for node in shuffled_edge])
    return sample_str

if __name__ == "__main__":
    args = get_args()
    args.graph_path = os.path.join(args.graph_dir, args.graph)
    
    if not os.path.exists(args.data_root_dir):
        os.mkdir(args.data_root_dir)
    args.data_dir = os.path.join(args.data_root_dir, f"{args.graph}_num_samples={args.num_samples}")
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    # Load model tokenizer
    args.model_path = os.path.join(args.hf_model_dir, args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token = tokenizer.sep_token
        
    G = nx.read_edgelist(args.graph_path)
    START_TOKEN_ID = tokenizer.convert_tokens_to_ids('a')
    vocab = [tokenizer.convert_ids_to_tokens(START_TOKEN_ID + i) for i in range(G.number_of_nodes())]
    for i in range(args.num_trials):
        data_path = os.path.join(args.data_dir, "samples_{}.pt".format(i))
        
        if os.path.exists(data_path):
            Warning(f"{data_path} already exists. Just continue.")
            continue
        
        sample_strs = generate_samples_from_graph(G, vocab, args.num_samples)
        # pdb.set_trace()
        torch.save(sample_strs, data_path)