import os
import argparse
import random
import networkx as nx
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="the number of ER graph nodes")
    parser.add_argument("--p", type=float, default=0.2)
    parser.add_argument("--min_weight", type=int, default=1)
    parser.add_argument("--max_weight", type=int, default=3)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--graph_dir", type=str, default="./graphs")
    
    return parser.parse_args()

def add_weights(G, args):
    weights = np.array([random.choice([args.min_weight, args.max_weight]) for _ in range(G.number_of_edges())])
    normalized_weights = weights / np.sum(weights)
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        d['weight'] = normalized_weights[i]

def generate_weighted_connected_gnm_random_graphs(args):
    m = int(args.n * (args.n - 1) / 2 * args.p)
    if not os.path.exists(args.graph_dir):
        os.mkdir(args.graph_dir)

    for i in range(args.num_trials):
        while True:
            G = nx.gnm_random_graph(n=args.n, m=m)
            if nx.is_connected(G):
                print("Generate WCGNM {}".format(i))
                add_weights(G, args)
                graph_path = os.path.join(args.graph_dir, f"WCGNM_n={args.n}_p={args.p}_kappa={args.max_weight / args.min_weight}_{i}")
                if os.path.exists(graph_path):
                    raise FileExistsError(f"{graph_path} already exists. Just continue.")
                nx.write_edgelist(G, graph_path)
                break

if __name__ == "__main__":
    args = get_args()
    generate_weighted_connected_gnm_random_graphs(args)
