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
import holoviews as hv
from holoviews import opts
import pandas as pd
import string

hv.extension("bokeh")
# hv.output(fig='svg', size=250)

color_palette = {
    'lightblue': '#ADD8E6', # light blue
    'lightred': '#FFCCCC',  # light red
    'lightgreen': '#CCFFCC',  # light green
    'lightyellow': '#FFFFCC',  # light yellow
    'lightpurple': '#CCCCFF',  # light purple
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--prompt_type", type=str, default="direct")
    parser.add_argument("--figure_save_dir", type=str, default="./figures")
    parser.add_argument("--source", type=str, help="the source concept from which the BFS generation starts")
    parser.add_argument("--max_depth", type=int, help="the maximum depth of the BFS generation")
    parser.add_argument("--max_width", type=int, help="the maximum width of the BFS generation")
    parser.add_argument("--color", type=str, default="lightblue")
    return parser.parse_args()
    

def abbr_word(word):
    if len(word) > 4:
        return word[:4] + "."
    else:
        return word


# def draw_concept_graph(adj_matrix, concepts, save_path=None):
#     G = nx.Graph(adj_matrix)
#     pos = nx.spring_layout(G, seed=123456)

#     # Nodes
#     nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue', alpha=0.9)

#     # Edges
#     nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='k')
    
#     # Labels
#     labels = {i: c for i, c in enumerate(concepts)}
#     # abbr_concepts = map(abbr_word, concepts)
#     # labels = {i: c for i, c in enumerate(abbr_concepts)}
#     nx.draw_networkx_labels(G, pos, labels, font_size=10, font_family='sans-serif')
#     plt.tight_layout()
#     plt.axis("off")
    
#     if save_path:
#         plt.savefig(save_path, dpi=300)
#     else:
#         plt.show()
    
def draw_multiple_graphs(adj_matrix_list, titles, concepts, color, set_title=True, info=None, save_path=None):
    G0 = nx.Graph(adj_matrix_list[0])
    # pos = nx.spring_layout(G0, seed=123456)
    pos = nx.circular_layout(G0)

    labels = {i: c[:2] + "." if len(c) > 2 else c for i, c in enumerate(concepts)}
    
    fig, axs = plt.subplots(1, len(adj_matrix_list), figsize=(24, 6))
    axs = axs.flat
    for i in range(len(adj_matrix_list)):
        adj_matrix = adj_matrix_list[i]
        title = titles[i]
        
        G = nx.Graph(adj_matrix)
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=axs[i], scale=1, scale_pos=1,
            node_size=3600,  # Size of nodes
            node_color=color_palette[color],  # Node color
            edgecolors='black',  # Node border color
        )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=axs[i], scale=1, scale_pos=1,
            alpha=0.5,  # Transparency of edges
            edge_color='grey'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos, labels, ax=axs[i], scale=1, scale_pos=1,
            font_size=12,  # Font size of labels
            font_weight='bold',  # Font weight of labels
            font_family='sans-serif'  # Font family
        )
        # nx.draw_networkx_labels(G, pos, labels, ax=axs[i])
        if set_title:
            axs[i].set_title(
                title,
                fontsize=18,  # Font size of labels
                fontweight='bold',  # Font weight of labels
                fontname='sans-serif'  # Font family
            )
        
    # Set the background color of the plot
    # plt.gca().set_facecolor('white')  # White background
    # plt.gca().collections[0].set_edgecolor("#000000")  # Black border for nodes

    # Remove axes
        axs[i].axis('off')
        
    for ax in axs:
        ax.margins(0.10)
    fig.tight_layout()
    
    # if info:
    #     plt.suptitle(info)
    
    if save_path:
        fig.savefig("{}.pdf".format(save_path), format="PDF", bbox_inches='tight', pad_inches=0)
    else:
        fig.show()
        
def draw_multiple_separate_graphs(adj_matrix_list, titles, concepts, color, set_title=True, info=None, save_path=None):
    G0 = nx.Graph(adj_matrix_list[0])
    pos = nx.circular_layout(G0)
    scale_factor = 0.83 # Value between 0 and 1, where 1 keeps nodes on the perimeter and 0 moves them to the center
    for node in pos:
        pos[node] = pos[node] * scale_factor

    # labels = {i: c[:2] + "." if len(c) > 2 else c for i, c in enumerate(concepts)}
    labels = {i: c for i, c in enumerate(concepts)}
    alphabet = list(string.ascii_uppercase)
    labels = {i: alphabet[i] for i, c in enumerate(concepts)}
    colors = plt.cm.viridis(np.linspace(0.3, 1, len(labels)))
    assert(len(labels) == len(concepts))
    assert(len(concepts) <= 12)
    # print("# Part 1")
    # part = concepts[:6]
    # tmps = [concepts[0]] + part + ["-"] * (6 - len(part))
    # tmps = " & ".join(tmps) + " \\\\"
    # print(tmps)

    # print("# Part 2")
    part = concepts[6:]
    tmps = [concepts[0]] + part + ["-"] * (6 - len(part))
    tmps = " & ".join(tmps) + " \\\\"
    print(tmps)
    
    # for i in range(len(adj_matrix_list)):
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     # fig, ax = plt.subplots()
    #     adj_matrix = adj_matrix_list[i]
    #     title = titles[i]
        
    #     G = nx.Graph(adj_matrix)
    #     # Draw nodes
    #     nx.draw_networkx_nodes(
    #         G, pos, ax=ax, 
    #         node_size=12000,  # Size of nodes
    #         node_color=colors, # color_palette[color],  # Node color
    #         edgecolors='black',  # Node border color
    #         linewidths=5
    #     )

    #     # Draw edges
    #     nx.draw_networkx_edges(
    #         G, pos, ax=ax, 
    #         alpha=0.6,  # Transparency of edges
    #         edge_color='grey',
    #         width=10.
    #     )
        
    #     # Draw labels
    #     nx.draw_networkx_labels(
    #         G, pos, labels, ax=ax,
    #         font_size=48,  # Font size of labels
    #         font_weight='bold',  # Font weight of labels
    #         font_family='sans-serif'  # Font family
    #     )
    #     # nx.draw_networkx_labels(G, pos, labels, ax=axs[i])
    #     if set_title:
    #         ax.set_title(
    #             title,
    #             fontsize=16,  # Font size of labels
    #             fontweight='bold',  # Font weight of labels
    #             fontname='sans-serif'  # Font family
    #         )
        
    # # Set the background color of the plot
    # # plt.gca().set_facecolor('white')  # White background
    # # plt.gca().collections[0].set_edgecolor("#000000")  # Black border for nodes

    # # Remove axes
    #     ax.axis('off')
    #     ax.set_xlim([-1, 1])
    #     ax.set_ylim([-1, 1])
    #     ax.set_aspect('equal')
    #     fig.tight_layout()
    
    # # if info:
    # #     plt.suptitle(info)
    
    #     if save_path:
    #         fig.savefig("{}_{}.pdf".format(save_path, title), format="PDF", bbox_inches='tight')
    #     else:
    #         fig.show()
            

if __name__ == "__main__":
    args = get_args()
    
    if not os.path.exists(args.figure_save_dir):
        os.mkdir(args.figure_save_dir)

    args.save_path = os.path.join(args.save_dir, "{}_max_depth={}_max_width={}.json".format(args.source, args.max_depth, args.max_width))   
    args.graph_save_path = os.path.join(args.save_dir, "{}_max_depth={}_max_width={}_graph.npy".format(args.source, args.max_depth, args.max_width))
    args.figure_save_path = os.path.join(args.figure_save_dir, "{}_max_depth={}_max_width={}_figure_{}_summary".format(args.source, args.max_depth, args.max_width, args.prompt_type))

    graph = np.load(args.graph_save_path, allow_pickle=True).item()
    models = ["llama-2-70b", "gpt-3.5-turbo-0613", "gpt-4-0613"]
    model_names = ["LLAMA-2-70B", "GPT-3.5", "GPT-4"]
    evaluated_graphs = []
    normalized_l1_ds = []
    
    for model in models:
        args.result_save_path = os.path.join(args.result_dir, "{}_max_depth={}_max_width={}_result_{}_{}.npy".format(args.source, args.max_depth, args.max_width, args.prompt_type, model))
        evaluated_graph = np.load(args.result_save_path, allow_pickle=True).item()
        l1_d = LA.norm(graph["adj_matrix"] - evaluated_graph["adj_matrix"], ord=1)
        normalized_l1_d = l1_d / LA.norm(graph["adj_matrix"], ord=1) 
        evaluated_graphs.append(evaluated_graph["adj_matrix"])
        normalized_l1_ds.append(normalized_l1_d)
    
    info = "{}".format(args.source)
    
    draw_multiple_separate_graphs(
        adj_matrix_list=[graph["adj_matrix"]] + evaluated_graphs,
        titles=["Ground Truth"] + model_names, 
        concepts=graph["i2c"], 
        color=args.color,
        set_title=False, # if args.source == "cake" else False,
        info=info,
        save_path=args.figure_save_path
    )
    # pdb.set_trace()
    # chord_diagram = draw_chord_diagram(adjacency_matrix=graph["adj_matrix"], labels=graph["i2c"])
    # hv.save(chord_diagram, 'chord_diagram.html', backend='bokeh')
    
    
        
    