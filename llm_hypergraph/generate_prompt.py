import os
import argparse
import numpy as np
import pandas as pd
import openai
from tqdm import tqdm

from utils import num_tokens_from_messages, merge_nested_dicts

# Configuration
def get_args():
    parser = argparse.ArgumentParser()
    
    # Problem hyperparameters
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--source", type=str, help="the source concept from which the BFS generation starts")
    parser.add_argument("--max_depth", type=int, help="the maximum depth of the BFS generation")
    parser.add_argument("--max_width", type=int, help="the maximum width of the BFS generation")
    
    # LLM hyperparameters
    parser.add_argument("--prompt_dir", type=str, default="./prompts")
    parser.add_argument("--prompt_type", type=str)
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--model", type=str, default="gpt-4-0613")
    return parser.parse_args()

# Completion
def generate_prompt_direct(concepts, args):
    prompt = "Consider the following concepts:"
    prompt = prompt + " " + concepts[0]
    for concept in concepts[1:]:
        prompt = prompt + ", " + concept
    prompt = prompt + ".\n"
    prompt = prompt + "Suppose that these concepts are nodes of a hypergraph.\n" 
    # prompt = prompt + "For each concept, consider {} most related concepts.\n".format(args.max_width)
    prompt = prompt + "According to the relations between these concepts, which hyperedges should be included? Please answer with a list of hyperedges."
    print(prompt)
    df = pd.DataFrame({"prompt": [prompt]})
    return df
    

if __name__ == "__main__":
    args = get_args()
    if args.prompt_type == "direct":
        args.num_prompts = 1
    
    if not os.path.exists(args.prompt_dir):
        os.mkdir(args.prompt_dir) 

    args.save_path = os.path.join(args.save_dir, "{}_max_depth={}_max_width={}.json".format(args.source, args.max_depth, args.max_width))   
    args.graph_save_path = os.path.join(args.save_dir, "{}_max_depth={}_max_width={}_graph.npy".format(args.source, args.max_depth, args.max_width))
    args.prompt_path = os.path.join(args.prompt_dir, 
                "{}_max_depth={}_max_width={}_prompt_{}.csv".format(args.source, args.max_depth, args.max_width,args.prompt_type))
    graph = np.load(args.graph_save_path, allow_pickle=True).item()
    if args.prompt_type == "direct":
        prompts = generate_prompt_direct(graph["i2c"], args)
        prompts.to_csv(args.prompt_path)

    
