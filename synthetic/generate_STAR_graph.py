import os
import argparse
import random
import networkx as nx
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="the number of ER graph nodes")
    parser.add_argument("--min_weight", type=int, default=1)
    parser.add_argument("--max_weight", type=int, default=3)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--graph_dir", type=str, default="./graphs")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for i in range(args.num_trials):
        weights = np.array([random.choice([args.min_weight, args.max_weight]) for _ in range(args.n - 1)])
        normalized_weights = weights / np.sum(weights)
        graph_path = os.path.join(args.graph_dir, f"STAR_n={args.n}_kappa={args.max_weight / args.min_weight}_{i}")
        if os.path.exists(graph_path):
            raise FileExistsError("File already exists.")
        with open(graph_path, "w") as f:
            for k in range(args.n-1):
                f.write("0 " + str(k+1) + " {'weight': " + str(normalized_weights[k]) + "}\n")
    
