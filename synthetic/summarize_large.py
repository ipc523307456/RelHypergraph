import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter1d
import pdb


eval_result_dir = "./eval_results_large"
colors = [
    "#0984e3", # Electron blue
    "#d63031", # Chi-gong
    "#00b894", # Mint leaf
    "#fdcb6e", # Bright yarrow
    "#6c5ce7", # Exodus fruit
    "#2d3436", # Darcula orchid
]

lighter_colors = {
    "#0984e3": "#74b9ff",
    "#d63031": "#ff7675",
    "#00b894": "#55efc4",
    "#fdcb6e": "#ffeaa7", 
    "#6c5ce7": "#a29bfe",
    "#2d3436": "#636e72", 
}

# Marker
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_']

def plot_eval_results_both(eval_result_dir, graph2eval_results, xlabel, ylabel, save_file, graph2label=None):
    ckpts = list(range(1000, 15000 + 1, 1000))
    plt.figure()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), constrained_layout=True)
    graphs = list(graph2eval_results.keys())
    n = len(graphs)
    eval_result_types = ["unweighted_dist", "weighted_dist"]
    titles = ["Unweighted", "Weighted"]
    
    # Set default font sizes
    # plt.rcParams.update({'font.size': 12})  # Adjust the size as needed
    # plt.rcParams.update({'axes.titlesize': 14})  # Adjust the title size as needed
    # plt.rcParams.update({'axes.labelsize': 12})  # Adjust the axis labels size as needed
    
    for i in range(2):
        ax = axs[i]
        eval_result_type = eval_result_types[i]
        title = titles[i]
        print(title)
        for graph, color, marker in zip(graphs, colors, markers):
            eval_result_files = graph2eval_results[graph]
            eval_results = []
            for eval_result_file in eval_result_files:
                file_path = os.path.join(eval_result_dir, f"{eval_result_file}.pth")
                eval_result = torch.load(file_path)[eval_result_type]
                eval_results.append(eval_result)
            # pdb.set_trace()
            if graph2label:
                plot_one(ax, ckpts, eval_results, label=graph2label[graph], color=color, marker=marker)
            else:
                plot_one(ax, ckpts, eval_results, label=graph, color=color, marker=marker)
        ax.set_xlabel(xlabel, fontsize="large")
        ax.set_ylabel(ylabel, fontsize="large")
        ax.set_title(title, fontsize="large")
        ax.legend(fontsize="large")
    
    plt.savefig(os.path.join(f"./figures_large/{save_file}.pdf"), bbox_inches='tight')

# def plot_eval_results(eval_result_dir, graph2eval_results, eval_result_type, xlabel, ylabel, title, save_file, graph2label=None):
#     ckpts = list(range(1000, 15000 + 1, 1000))
#     plt.figure()
#     fig, ax = plt.subplots()
#     graphs = list(graph2eval_results.keys())
#     n = len(graphs)
    
#     for graph, color, marker in zip(graphs, colors, markers):
#         eval_result_files = graph2eval_results[graph]
#         eval_results = []
#         for eval_result_file in eval_result_files:
#             file_path = os.path.join(eval_result_dir, f"{eval_result_file}.pth")
#             eval_result = torch.load(file_path)[eval_result_type]
#             eval_results.append(eval_result)
#         # pdb.set_trace()
#         if graph2label:
#             plot_one(ax, ckpts, eval_results, label=graph2label[graph], color=color, marker=marker)
#         else:
#             plot_one(ax, ckpts, eval_results, label=graph, color=color, marker=marker)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     if title:
#         ax.set_title(title)
#     ax.legend()
    
#     plt.savefig(os.path.join(f"./figures/{save_file}.pdf"), bbox_inches='tight')


def plot_one(ax, ckpts, eval_results, label, color, marker):

    mean_eval_result = np.mean([eval_result for eval_result in eval_results], axis=0)
    std_eval_result = np.std([eval_result for eval_result in eval_results], axis=0)
    std_eval_result = gaussian_filter1d(std_eval_result, sigma=1)
    print(label)
    print([mval for mval, ckpt in zip(mean_eval_result, ckpts) if ckpt % 3000 == 0])
    print(ckpts)
    print(mean_eval_result)
    ax.plot(ckpts, mean_eval_result, label=label, color=color, marker=marker)
    # ax.fill_between(ckpts, mean_eval_result - std_eval_result, mean_eval_result + std_eval_result, color=color, alpha=0.2)
    ax.errorbar(ckpts, mean_eval_result, yerr=std_eval_result, fmt='--', color=color, ecolor=lighter_colors[color], elinewidth=1, capsize=3)

# Kappa
# ckpts = list(range(1000, 15000 + 1, 1000))
ckpts = list(range(1000, 15000 + 1, 1000))
graphs2eval_results = {
    "n=10": [
        "graph=WCGNM_n=10_p=0.2_kappa=3.0_0_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=10_p=0.2_kappa=3.0_1_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=10_p=0.2_kappa=3.0_2_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=10_p=0.2_kappa=3.0_3_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=10_p=0.2_kappa=3.0_4_training_arguments=training_arguments_1_trial=0",
    ],
    "n=20": [
        "graph=WCGNM_n=20_p=0.2_kappa=3.0_0_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=20_p=0.2_kappa=3.0_1_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=20_p=0.2_kappa=3.0_2_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=20_p=0.2_kappa=3.0_3_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=20_p=0.2_kappa=3.0_4_training_arguments=training_arguments_1_trial=0",
    ],
    "n=50": [
        "graph=WCGNM_n=50_p=0.2_kappa=3.0_0_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=50_p=0.2_kappa=3.0_1_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=50_p=0.2_kappa=3.0_2_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=50_p=0.2_kappa=3.0_3_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=50_p=0.2_kappa=3.0_4_training_arguments=training_arguments_1_trial=0",
    ],
    "n=100": [
        "graph=WCGNM_n=100_p=0.2_kappa=3.0_0_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=100_p=0.2_kappa=3.0_1_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=100_p=0.2_kappa=3.0_2_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=100_p=0.2_kappa=3.0_3_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=100_p=0.2_kappa=3.0_4_training_arguments=training_arguments_1_trial=0",
    ],
    "n=200": [
        "graph=WCGNM_n=200_p=0.2_kappa=3.0_0_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=200_p=0.2_kappa=3.0_1_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=200_p=0.2_kappa=3.0_2_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=200_p=0.2_kappa=3.0_3_training_arguments=training_arguments_1_trial=0",
        "graph=WCGNM_n=200_p=0.2_kappa=3.0_4_training_arguments=training_arguments_1_trial=0",
    ],
}

graph2label = {
    "n=10": r"WCGNM ($n=10$)",
    "n=20": r"WCGNM ($n=20$)",
    "n=50": r"WCGNM ($n=50$)",
    "n=100": r"WCGNM ($n=100$)",
    "n=200": r"WCGNM ($n=200$)",
}

plot_eval_results_both(eval_result_dir, graphs2eval_results,
        xlabel="Step",
        ylabel=r"$\ell_1$ Distance",
        save_file="Different_num_nodes",
        graph2label=graph2label,
    )

