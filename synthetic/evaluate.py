import os
import argparse
import string
import numpy as np
import numpy.linalg as LA
from queue import Queue
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import networkx as nx
from utils import l1_distance
from itertools import permutations
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import time
from pprint import pprint
from copy import deepcopy
import pdb

from data import create_data_collator
from preprocess import preprocess_function

# threshold = 1E-3
threshold = 1E-2

def pprint_array(arr):
    m, n = arr.shape
    format_str = "".join(["{:.3f} "] * n)
    for i in range(m):
        row = [float(arr[i][j]) for j in range(n)]
        print(format_str.format(*row))

def get_probability_from_model(prefix_str, tokenizer, model, node_ids, args, exclude_node_ids=None):    
    """
        Adapted from https://github.com/benpry/why-think-step-by-step
    """
    
    token_ids = tokenizer(prefix_str, return_tensors="pt").input_ids
    token_ids = torch.cat([101 * torch.ones((token_ids.shape[0], 1), dtype=torch.long), token_ids[:,1:-1], 
        103 * torch.ones((token_ids.shape[0], 1), dtype=torch.long), 102 * torch.ones((token_ids.shape[0], 1), dtype=torch.long)], dim=1).to("cuda")
    output_logits = model(token_ids).logits.detach().cpu()
    node_logits = output_logits[0][-2][node_ids]
    node_prob = torch.softmax(node_logits, dim=-1)
    if exclude_node_ids:
        exclude_nodes = [id - args.START_TOKEN_ID for id in exclude_node_ids]
        node_prob[exclude_nodes] = 0
        node_prob = node_prob / torch.sum(node_prob)
    
    return node_prob

def bfs_weight_estimation(relative_weight_matrix, mask):
    unormalized_weight_matrix = np.zeros_like(relative_weight_matrix)
    visited = np.zeros_like(relative_weight_matrix)
    n = unormalized_weight_matrix.shape[0]
    init_edge = None
    for i in range(1, n):
        if mask[0][i] > threshold:
            init_edge = (0, i)
            break
    unormalized_weight_matrix[init_edge[0]][init_edge[1]] = 1
    visited[init_edge[0]][init_edge[1]] = 1
    
    Q = Queue(maxsize=n*n)
    Q.put(init_edge)
    while not Q.empty():
        edge = Q.get()
        x, y = edge
        
        for j in range(n): 
            if j == y:
                continue
            
            a, b =  min(x, j), max(x, j)
            
            if mask[a][b] > 0:
                if visited[a][b] > 1E-5:
                    continue
                
                # if unormalized_weight_matrix[a][b] > 1E-5:
                #     continue
                
                unormalized_weight_matrix[a][b] = unormalized_weight_matrix[x][y] * relative_weight_matrix[x][j] / relative_weight_matrix[x][y]
                visited[a][b] = 1
                Q.put((a, b))
        
        for i in range(n): 
            if i == x:
                continue
            
            a, b =  min(y, i), max(y, i)
            
            if mask[a][b] > 0:
                if visited[a][b] > 1E-5:
                    continue
                
                # if unormalized_weight_matrix[a][b] > 1E-5:
                #     continue
                
                unormalized_weight_matrix[a][b] = unormalized_weight_matrix[x][y] * relative_weight_matrix[y][i] / relative_weight_matrix[y][x]
                visited[a][b] = 1
                Q.put((a, b))
    # print("RWM")
    # print(relative_weight_matrix)
    # print("UWM")
    # print(unormalized_weight_matrix)
    # print("NWM")
    normalized_weight_matrix = unormalized_weight_matrix / np.sum(unormalized_weight_matrix)
    # print(normalized_weight_matrix)
    normalized_weight_matrix = normalized_weight_matrix + normalized_weight_matrix.T
    return normalized_weight_matrix
    

def estimate_graph(tokenizer, model, node_ids, args):
    num_nodes = len(node_ids)
    print("t1")
    t1 = time.time()
    
    edge_probs = []
    for i in tqdm(range(num_nodes)):
        node_id = i+args.START_TOKEN_ID
        node = args.id2token(node_id)
        edge_prob = get_probability_from_model(
            prefix_str="{}".format(node),
            tokenizer=tokenizer,
            model=model,
            node_ids=node_ids,
            args=args,
            exclude_node_ids=[node_id],
        )
        # edge_prob = edge_prob * node_probs[i]
        edge_probs.append(edge_prob)
    # multi_factors = [(edge_probs[0][i] / edge_probs[i][0]) for i in range(1, num_nodes)]
    # for i in range(1, num_nodes):
    #     edge_probs[i] = edge_probs[i] * multi_factors[i - 1]
    edge_probs = torch.stack(edge_probs, dim=0)
    # edge_probs = edge_probs / torch.sum(edge_probs)
    edge_probs = edge_probs.numpy()
    # pdb.set_trace()
    # edge_probs = edge_probs / np.max(edge_probs, axis=1).reshape((-1,1))
    # pdb.set_trace()
    edge_probs_mask = deepcopy(edge_probs)
    edge_probs_mask[edge_probs < threshold] = 0
    edge_probs_mask[edge_probs > threshold] = 1
    adj_matrix = edge_probs_mask
    # pdb.set_trace()
    print("t2")
    t2 = time.time()
    weighted_adj_matrix = bfs_weight_estimation(edge_probs, np.array(edge_probs_mask, dtype=np.bool_))
    t3 = time.time()

    return adj_matrix, weighted_adj_matrix
    
def get_args():
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument("--graph_dir", type=str, default="./graphs")
    parser.add_argument("--graph", type=str)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--trial", type=int)
    
    # Model
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--hf_model_dir", type=str, default="./hf_models")
    parser.add_argument("--init", type=str, default="scratch")
    
    # Trainer's Hyperparameters
    parser.add_argument("--output_root_dir", type=str, default="./results")
    parser.add_argument("--training_arguments_dir", type=str, default="./training_arguments")
    parser.add_argument("--training_arguments", type=str)
    
    # Eval Results
    parser.add_argument("--eval_result_dir", type=str, default="./eval_results")
    
    parser.add_argument("--savefig_root_dir", type=str, default="./figures")
    
    return parser.parse_args()

def permute_adj_matrix(adj_matrix, permutation):
    intermediate_matrix = adj_matrix[permutation, :]
    permuted_matrix = intermediate_matrix[:, permutation]   
    return permuted_matrix

def are_isomorphic(matrix1, matrix2):
    n = matrix1.shape[0]
    min_distance = 999999
    
    all_permutations = list(permutations(range(n)))
    for p in all_permutations:
        permuted_matrix2 = permute_adj_matrix(matrix2, p)
        tmp = l1_distance(matrix1, permuted_matrix2)
        if tmp < min_distance:
            min_distance = tmp 
    return min_distance

def is_automorphic(matrix):
    n = matrix.shape[0]
    min_distance = 999999
    
    all_permutations = list(permutations(range(n)))
    for p in all_permutations[1:]:
        permuted_matrix = permute_adj_matrix(matrix, p)
        tmp = l1_distance(matrix, permuted_matrix)
        if tmp < min_distance:
            min_distance = tmp 
    return min_distance

if __name__ == "__main__":
    # A = [[0, 1, 2], [0, 0, 3], [0, 0, 0]]   
    # A = np.array(A)
    # A = A + A.T
    # nA = A / np.sum(A) * 2
    # rA = A
    # mrA = A / np.max(A, axis=1).reshape((-1,1))
    # mask = (rA > 1E-6)
    # eA = bfs_weight_estimation(rA, mask)
    # print("rA")
    # print(rA)
    # print("nA")
    # print(nA)
    # print("eA")
    # print(eA)
    # pdb.set_trace()
    
    args = get_args()
    
    args_training_arguments = None
    with open(os.path.join(args.training_arguments_dir, f"{args.training_arguments}.yaml"), "r") as f:
        args_training_arguments = yaml.safe_load(f)
    
    args.eval_result_path = os.path.join(args.eval_result_dir, f"graph={args.graph}_training_arguments={args.training_arguments}_trial={args.trial}.pth")
    if os.path.exists(args.eval_result_path):
        eval_results = torch.load(args.eval_result_path)
        ckpts = eval_results["ckpts"]
        unweighted_dist = eval_results["unweighted_dist"]
        weighted_dist = eval_results["weighted_dist"]
    else:
        # Load model tokenizer
        print("Load tokenizer")
        args.output_dir = os.path.join(args.output_root_dir, 
            f"{args.model}_{args.init}_graph={args.graph}_training_arguments={args.training_arguments}_trial_{args.trial}")
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.hf_model_dir, args.model))
        tokenizer.pad_token = tokenizer.eos_token = tokenizer.sep_token
        
        args.START_TOKEN_ID = tokenizer.convert_tokens_to_ids('a')
        args.id2token = tokenizer.convert_ids_to_tokens

        # Load graph
        print("Load graph")
        G = nx.read_edgelist(os.path.join(args.graph_dir, args.graph))
        permutation = None
        with open(os.path.join(args.graph_dir, args.graph)) as f:
            lines = f.readlines()
            nodes = []
            for line in lines:
                nodes.extend(line.split()[:2])
            nodes = [int(node) for node in nodes]
            nodes = list(dict.fromkeys(nodes).keys())
            # print(nodes)
        permutation = np.array(list(range(len(nodes))))
        permutation[nodes] = np.array(list(range(len(nodes))))
        # print(permutation)
        G_adj_matrix =  permute_adj_matrix(nx.to_numpy_array(G), permutation)
        weighted_G_adj_matrix = permute_adj_matrix(nx.to_numpy_array(G), permutation)
        # print(weighted_G_adj_matrix)
        G_adj_matrix[G_adj_matrix < threshold] = 0
        G_adj_matrix[G_adj_matrix > threshold] = 1
        node_ids = [args.START_TOKEN_ID + i for i in range(G.number_of_nodes())]

        ckpts = list(range(1000, 15000+1, 1000))
        unweighted_dist = []
        weighted_dist = []
        estimated_G_adj_matrices = []
        estimated_weighted_G_adj_matrices = []  
        # print("G")
        # wG = weighted_G_adj_matrix / np.max(weighted_G_adj_matrix, axis=1).reshape((-1,1))
        # print(wG)
        # eG = bfs_weight_estimation(wG, G_adj_matrix)
        # pdb.set_trace()

        for ckpt in tqdm(ckpts):
            print("ckpt")
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-{ckpt}")
            print("load model before")
            model = AutoModelForMaskedLM.from_pretrained(ckpt_path)
            print("to cuda")
            model = model.to("cuda")
            print("load model after")
            estimated_G_adj_matrix, estimated_weighted_G_adj_matrix = estimate_graph(tokenizer=tokenizer, model=model, node_ids=node_ids, args=args)
            estimated_G_adj_matrices.append(estimated_G_adj_matrix)
            estimated_weighted_G_adj_matrices.append(estimated_weighted_G_adj_matrix)
            uwd = l1_distance(G_adj_matrix, estimated_G_adj_matrix)/2
            wd = l1_distance(weighted_G_adj_matrix, estimated_weighted_G_adj_matrix)/2
            unweighted_dist.append(uwd)
            weighted_dist.append(wd)
            del model

        
        eval_results = {}
        eval_results["ckpts"] = ckpts
        eval_results["G_adj_matrix"] = G_adj_matrix
        eval_results["estimated_G_adj_matrices"] = estimated_G_adj_matrices
        eval_results["estimated_weighted_G_adj_matrices"] = estimated_weighted_G_adj_matrices
        eval_results["unweighted_dist"] = unweighted_dist
        eval_results["weighted_dist"] = weighted_dist
        
        # pdb.set_trace()
        
        if not os.path.exists(args.eval_result_dir):
            os.mkdir(args.eval_result_dir)
        torch.save(eval_results, args.eval_result_path)
        
    
    if not os.path.exists(args.savefig_root_dir):
        os.mkdir(args.savefig_root_dir)
    
    args.savefig_dir = os.path.join(args.savefig_root_dir, f"{args.graph}_trial_{args.trial}")
    if not os.path.exists(args.savefig_dir):
        os.mkdir(args.savefig_dir)
        
    print(unweighted_dist)
    print(weighted_dist)
    
    plt.figure()
    plt.title(f"Unweighted ({args.graph})")
    plt.xlabel("Training Steps")
    plt.ylabel("L1 Distance")
    plt.plot(ckpts, unweighted_dist)
    plt.savefig(os.path.join(args.savefig_dir, f"{args.graph}_unweighted.pdf"))
    
    plt.figure()
    plt.title(f"Weighted ({args.graph})")
    plt.xlabel("Training Steps")
    plt.ylabel("L1 Distance")
    plt.plot(ckpts, weighted_dist)
    plt.savefig(os.path.join(args.savefig_dir, f"{args.graph}_weighted.pdf"))
    