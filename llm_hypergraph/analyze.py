import os
import re
import argparse
import numpy as np
import pandas as pd
import openai
from tqdm import tqdm

from utils import num_tokens_from_messages, merge_nested_dicts
import pdb


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
    parser.add_argument("--response_dir", type=str, default="./responses")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--model", type=str, default="gpt-4-0613")
    return parser.parse_args()

# Analyze responses
def extract_wrapped_content(input_string):
    pattern = r"\{[^}]*\}|\[[^\]]*\]|\([^)]*\)"
    return re.findall(pattern, input_string)

def find_all_concepts(content):
    extracted_content = extract_wrapped_content(content)
    
    if len(extracted_content) > 0:
        concepts = re.findall(r"\b[A-Za-z]+\b", extracted_content[0])
        return concepts
    else:
        return None

def text2hypergraph(content):
    hypergraph = {}
    i = 0
    for line in content.split("\n"):
        hyperedge = find_all_concepts(line)
        if hyperedge:
            hypergraph[f"e{i}"] = hyperedge
            i = i + 1
    return hypergraph

def analyze_direct(response, graph, args):
    res = graph
    c2i = graph["c2i"]
    content = response.at[0, "choices_message"]
    hypergraph = text2hypergraph(content)
    for k, v in hypergraph.items():
        try:
            hypergraph[k] = [c2i[concept] for concept in v]
        except KeyError:
            Warning("Unexpected concepts in the hyperedge.")
            hypergraph[k] = []
    res["evaluated_hypergraph"] = hypergraph
    np.save(args.result_path, res, allow_pickle=True)

if __name__ == "__main__":
    args = get_args()
    if args.prompt_type == "direct":
        args.num_prompts = 1
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir) 

    args.save_path = os.path.join(args.save_dir, "{}_max_depth={}_max_width={}.json".format(args.source, args.max_depth, args.max_width))   
    args.graph_save_path = os.path.join(args.save_dir, "{}_max_depth={}_max_width={}_graph.npy".format(args.source, args.max_depth, args.max_width))
    args.prompt_path = os.path.join(args.prompt_dir, 
                "{}_max_depth={}_max_width={}_prompt_{}.csv".format(args.source, args.max_depth, args.max_width,args.prompt_type))
    args.response_path = os.path.join(args.response_dir, 
                "{}_max_depth={}_max_width={}_response_{}_{}.csv".format(args.source, args.max_depth, args.max_width,args.prompt_type, args.model))
    args.result_path = os.path.join(args.result_dir, 
                "{}_max_depth={}_max_width={}_result_{}_{}.npy".format(args.source, args.max_depth, args.max_width,args.prompt_type, args.model))
    prompts = pd.read_csv(args.prompt_path)
    response = pd.read_csv(args.response_path)
    
    graph = np.load(args.graph_save_path, allow_pickle=True).item()
    if args.prompt_type == "direct":
        analyze_direct(response, graph, args)
    
    
