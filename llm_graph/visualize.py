import os
import argparse
import requests
import json
import time
from collections import deque
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import networkx as nx
import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--prompt_type", type=str, default="direct")
    parser.add_argument("--figure_save_dir", type=str, default="./figures")
    parser.add_argument("--source", type=str, help="the source concept from which the BFS generation starts")
    parser.add_argument("--max_depth", type=int, help="the maximum depth of the BFS generation")
    parser.add_argument("--max_width", type=int, help="the maximum width of the BFS generation")
    parser.add_argument("--model", type=str, default="gpt-4-0613")
    return parser.parse_args()
    

def abbr_word(word):
    if len(word) > 4:
        return word[:4] + "."
    else:
        return word
    

    

def draw_concept_graph(adj_matrix, concepts, save_path=None):
    G = nx.Graph(adj_matrix)
    pos = nx.spring_layout(G, seed=123456)
    options = {
        "node_size": 1000,
    }
    nx.draw(G, pos, **options)
    labels = {i: c for i, c in enumerate(concepts)}
    # abbr_concepts = map(abbr_word, concepts)
    # labels = {i: c for i, c in enumerate(abbr_concepts)}
    nx.draw_networkx_labels(G, pos, labels)
    plt.tight_layout()
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
def draw_multiple_graphs(adj_matrix_list, titles, concepts, info=None, save_path=None):
    G0 = nx.Graph(adj_matrix_list[0])
    # pos = nx.spring_layout(G0, seed=123456)
    pos = nx.shell_layout(G0)
    options = {
        "node_size": 500,
    }
    labels = {i: c for i, c in enumerate(concepts)}
    
    fig, axs = plt.subplots(1, len(adj_matrix_list), figsize=(12, 6))
    axs = axs.flat
    for i in range(len(adj_matrix_list)):
        adj_matrix = adj_matrix_list[i]
        title = titles[i]
        
        G = nx.Graph(adj_matrix)
        nx.draw(G, pos, ax=axs[i], **options)
        nx.draw_networkx_labels(G, pos, labels, ax=axs[i])
        axs[i].set_title(title)
        
    for ax in axs:
        ax.margins(0.10)
    # fig.tight_layout()
    
    if info:
        plt.suptitle(info)
    
    if save_path:
        fig.savefig(save_path)
    else:
        fig.show()
        
        


if __name__ == "__main__":
    args = get_args()
    
    if not os.path.exists(args.figure_save_dir):
        os.mkdir(args.figure_save_dir)

    args.save_path = os.path.join(args.save_dir, "{}_max_depth={}_max_width={}.json".format(args.source, args.max_depth, args.max_width))   
    args.graph_save_path = os.path.join(args.save_dir, "{}_max_depth={}_max_width={}_graph.npy".format(args.source, args.max_depth, args.max_width))
    args.figure_save_path = os.path.join(args.figure_save_dir, "{}_max_depth={}_max_width={}_figure_{}_{}.png".format(args.source, args.max_depth, args.max_width, args.prompt_type, args.model))
    args.result_save_path = os.path.join(args.result_dir, "{}_max_depth={}_max_width={}_result_{}_{}.npy".format(args.source, args.max_depth, args.max_width, args.prompt_type, args.model))

    graph = np.load(args.graph_save_path, allow_pickle=True).item()
    evaluated_graph = np.load(args.result_save_path, allow_pickle=True).item()
    l1_d = LA.norm(graph["adj_matrix"] - evaluated_graph["adj_matrix"], ord=1)
    normalized_l1_d = l1_d / LA.norm(graph["adj_matrix"], ord=1) 
    
    info = "{} (Normalized L1 = {})".format(args.source, normalized_l1_d)
    
    draw_multiple_graphs(
        adj_matrix_list=[graph["adj_matrix"], evaluated_graph["adj_matrix"]],
        titles=["ground truth", "{} ({})".format(args.model, args.prompt_type)], 
        concepts=graph["i2c"], 
        info=info,
        save_path=args.figure_save_path
    )
    
    
    
    
    
    
        
    