import os
import argparse
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import pdb

from utils import merge_nested_dicts

client = OpenAI()

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
    parser.add_argument("--model", type=str, default="gpt-4-0613")
    return parser.parse_args()

# Completion
def get_response(model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    response = vars(response)
    response["choices"] = vars(response["choices"][0])
    response["choices"]["message"] = response["choices"]["message"].content
    
    return response

def save_response(cur_response, prev_response_df, args):
    cur_response_dict = merge_nested_dicts([cur_response])
    cur_response_df = pd.DataFrame(cur_response_dict)
    merged_response_df = pd.concat([prev_response_df, cur_response_df], ignore_index=True)
    merged_response_df.to_csv(args.response_path)
    return merged_response_df


if __name__ == "__main__":
    args = get_args()
    if args.prompt_type == "direct":
        args.num_prompts = 1
    
    if not os.path.exists(args.response_dir):
        os.mkdir(args.response_dir) 

    args.prompt_path = os.path.join(args.prompt_dir, 
                "{}_max_depth={}_max_width={}_prompt_{}.csv".format(args.source, args.max_depth, args.max_width,args.prompt_type, args.model))
    args.response_path = os.path.join(args.response_dir, 
                "{}_max_depth={}_max_width={}_response_{}_{}.csv".format(args.source, args.max_depth, args.max_width,args.prompt_type, args.model))
    prompts = pd.read_csv(args.prompt_path)
    
    responses_df = None
    num_existed = 0 
    if os.path.exists(args.response_path):
        responses_df = pd.read_csv(args.response_path)
        num_existed = responses_df.shape[0]
    
    for i in tqdm(range(num_existed, args.num_prompts)):
        prompt = prompts.at[i, "prompt"]
        response = get_response(model=args.model, prompt=prompt)
        responses_df = save_response(cur_response=response, prev_response_df=responses_df, args=args)
    
    
