import os
import re
import argparse
import numpy as np
import pandas as pd
import openai
from tqdm import tqdm

from utils import num_tokens_from_messages, merge_nested_dicts
import pdb

# Authentication
openai.organization = "org-9qVdahbq4vvR45xO8CHD9Oym"
openai.api_key = os.getenv("OPENAI_API_KEY2")

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
def find_all_concepts(content):
    concepts = re.findall(r"\b[A-Za-z]+\b", content)
    return concepts

def analyze_direct(response, graph, args):
    res = graph
    c2i = graph["c2i"]
    res["adj_matrix"] = np.zeros_like(graph["adj_matrix"])
    content = response.at[0, "choices_message_content"]
    # concepts = content.split('\n')
    # concepts = [(c.split('.')[1][1:] if '.' in c else c) for c in concepts]
    # concepts = [c.split(', ') for c in concepts]
    # lhs = [c[0][1:] for c in concepts]
    # rhs = [c[1][:-1] for c in concepts]
    concepts = find_all_concepts(content)
    # pdb.set_trace()
    num_concepts = graph["num_concepts"]
    # pdb.set_trace()
    # assert(num_concepts * args.max_width * 2 >= len(concepts))
    assert(len(concepts) % 2 == 0)
    lhs = [concepts[2 * i].lower() for i in range(len(concepts)//2)]
    rhs = [concepts[2 * i + 1].lower() for i in range(len(concepts)//2)]
    for l, r in zip(lhs, rhs):
        try:
            res["adj_matrix"][c2i[l]][c2i[r]] = 1
            res["adj_matrix"][c2i[r]][c2i[l]] = 1
        except KeyError:
            continue
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
    
    
