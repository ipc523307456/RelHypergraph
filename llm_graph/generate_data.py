#
# Generate related concepts from ConceptNet
#

import os
import argparse
import requests
import json
import time
from collections import deque
from utils import normalize_word, remove_duplicates

class ConceptNet:
    def __init__(self):
        pass
    
    @staticmethod
    def query(concept):
        content = requests.get("http://api.conceptnet.io/c/en/{}".format(concept))
        try:
            edges = content.json()["edges"]
        except json.decoder.JSONDecodeError:
            print(content)
            return []
        
        ans = [(edge["rel"]["label"], edge["end"]["label"]) for edge in edges if edge["end"]["language"] == "en"]
        return ans
    
    @staticmethod
    def rquery(concept, max_width=None, valid_concepts=None):
        content = requests.get("http://api.conceptnet.io/c/en/{}".format(concept))
        try:
            edges = content.json()["edges"]
        except json.decoder.JSONDecodeError:
            print(content)
            return []
        
        time.sleep(1) # to avoid exceeding the rate limits of ConceptNet Web API
        related_concepts = []        
        related_concepts.extend([(edge["end"]["label"].split()[-1], edge["weight"]) for edge in edges if (edge["end"]["language"] == "en") and (concept in edge["start"]["label"]) and (not concept in edge["end"]["label"])])
        related_concepts.extend([(edge["start"]["label"].split()[-1], edge["weight"]) for edge in edges if (edge["start"]["language"] == "en") and (not concept in edge["start"]["label"]) and (concept in edge["end"]["label"])])
        sorted(related_concepts, key=lambda a: a[1], reverse=True)
        related_concepts = [normalize_word(c[0]) for c in related_concepts] 
        related_concepts = remove_duplicates(related_concepts)
        
        if valid_concepts:
            related_concepts = [c for c in related_concepts if c in valid_concepts]
        
        if max_width:
            return related_concepts if len(related_concepts) < max_width else related_concepts[:max_width]
        else:
            return related_concepts           

def bfs_generation(graph, src, max_depth, max_width=None, valid_concepts=None):
    visited = set()
    queue = deque([(src, 0)])  
    res = {}
    all_concepts = []

    while queue:
        node, depth = queue.popleft()
        
        if node in visited:
            continue
        
        print(node)
        
        visited.add(node)        
        neighbors = graph.rquery(node, max_width, valid_concepts)
        res[node] = neighbors
        
        if depth >= max_depth:
            continue

        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))
        
    return res, visited

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--source", type=str, help="the source concept from which the BFS generation starts")
    parser.add_argument("--max_depth", type=int, help="the maximum depth of the BFS generation")
    parser.add_argument("--max_width", type=int, help="the maximum width of the BFS generation")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    _, valid_concepts = bfs_generation(graph=ConceptNet, src=args.source, max_depth=args.max_depth, max_width=args.max_width)
    related_concepts, _ = bfs_generation(graph=ConceptNet, src=args.source, max_depth=args.max_depth, max_width=args.max_width, valid_concepts=valid_concepts)
    assert(len(valid_concepts) == len(related_concepts))
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    args.save_path = os.path.join(args.save_dir, "{}_max_depth={}_max_width={}.json".format(args.source, args.max_depth, args.max_width))   
    with open(args.save_path, 'w') as f:
        json.dump(related_concepts, f)
    
        
    