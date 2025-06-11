import os
import argparse
import requests
import json
import time
from collections import deque
import numpy as np
from utils import normalize_word, remove_duplicates
import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--source", type=str, help="the source concept from which the BFS generation starts")
    parser.add_argument("--max_depth", type=int, help="the maximum depth of the BFS generation")
    parser.add_argument("--max_width", type=int, help="the maximum width of the BFS generation")
    return parser.parse_args()

def generate_concept_to_index_mapping(concepts_dict):
    concept_list = []
    for k, v in concepts_dict.items():
        if not k in concept_list:
            concept_list.append(k)
        for x in v:
            if not x in concept_list:
                concept_list.append(x)
    
    concept2index = {}
    index2concept = concept_list
    for i, c in enumerate(concept_list):
        concept2index[c] = i
        
    return concept2index, index2concept, len(concept_list)
    
def concepts_to_graph(concept_dict):        
    c2i, i2c, num_concepts = generate_concept_to_index_mapping(concept_dict)
    adj_matrix = np.zeros((num_concepts, num_concepts))
    for k, v in concept_dict.items():
        for x in v:
            adj_matrix[c2i[k]][c2i[x]] = 1
            adj_matrix[c2i[x]][c2i[k]] = 1
    return adj_matrix, c2i, i2c, num_concepts
    

if __name__ == "__main__":
    args = get_args()

    args.save_path = os.path.join(args.save_dir, "{}_max_depth={}_max_width={}.json".format(args.source, args.max_depth, args.max_width))   
    
    with open(args.save_path, 'r') as f:
        concepts = json.load(f)
    
    adj_matrix, c2i, i2c, num_concepts = concepts_to_graph(concepts)
    
    graph = {}
    graph["adj_matrix"] = adj_matrix
    graph["c2i"] = c2i
    graph["i2c"] = i2c
    graph["num_concepts"] = num_concepts
    
    args.graph_save_path = os.path.join(args.save_dir, "{}_max_depth={}_max_width={}_graph.npy".format(args.source, args.max_depth, args.max_width))
    np.save(args.graph_save_path, graph, allow_pickle=True)
    
    
    
    
    
    
        
    