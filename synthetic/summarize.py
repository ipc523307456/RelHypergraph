import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter1d
import pdb


eval_result_dir = "./eval_results"

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
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), constrained_layout=True)
    graphs = list(graph2eval_results.keys())
    n = len(graphs)
    eval_result_types = ["unweighted_dist", "weighted_dist"]
    titles = ["Unweighted", "Weighted"]
    
    # Set default font sizes
    # plt.rcParams.update({'font.size': 12})  # Adjust the size as needed
    # plt.rcParams.update({'axes.titlesize': 14})  # Adjust the title size as needed
    # plt.rcParams.update({'axes.labelsize': 12})  # Adjust the axis labels size as needed
    
    for i in range(2):
        eval_result_type = eval_result_types[i]
        title = titles[i]
        for graph, color, marker in zip(graphs, colors, markers):
            eval_result_files = graph2eval_results[graph]
            eval_results = []
            for eval_result_file in eval_result_files:
                file_path = os.path.join(eval_result_dir, f"{eval_result_file}.pth")
                eval_result = torch.load(file_path)[eval_result_type]
                eval_results.append(eval_result)
            # pdb.set_trace()
            if graph2label:
                plot_one(axs[i], ckpts, eval_results, label=graph2label[graph], color=color, marker=marker)
            else:
                plot_one(axs[i], ckpts, eval_results, label=graph, color=color, marker=marker)
        axs[i].set_xlabel(xlabel, fontsize="large")
        axs[i].set_ylabel(ylabel, fontsize="large")
        axs[i].set_title(title, fontsize="large")
        axs[i].legend(fontsize="large")
    
    plt.savefig(os.path.join(f"./figures/{save_file}.pdf"), bbox_inches='tight')

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
    ax.plot(ckpts, mean_eval_result, label=label, color=color, marker=marker)
    # ax.fill_between(ckpts, mean_eval_result - std_eval_result, mean_eval_result + std_eval_result, color=color, alpha=0.2)
    ax.errorbar(ckpts, mean_eval_result, yerr=std_eval_result, fmt='--', color=color, ecolor=lighter_colors[color], elinewidth=1, capsize=3)

# Kappa
ckpts = list(range(1000, 15000 + 1, 1000))
graphs2eval_results = {
    "kappa=1": [
        "graph=STAR_n=20_kappa=1.0_0_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=1.0_1_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=1.0_2_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=1.0_3_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=1.0_4_training_arguments=training_arguments_1_trial=0",
    ],
    "kappa=2": [
        "graph=STAR_n=20_kappa=2.0_0_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=2.0_1_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=2.0_2_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=2.0_3_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=2.0_4_training_arguments=training_arguments_1_trial=0",
    ],
    "kappa=3": [
        "graph=STAR_n=20_kappa=3.0_0_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=3.0_1_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=3.0_2_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=3.0_3_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=3.0_4_training_arguments=training_arguments_1_trial=0",
    ],
}

graph2label = {
    "kappa=1": r"STAR ($\kappa=1.0$)",
    "kappa=2": r"STAR ($\kappa=2.0$)",
    "kappa=3": r"STAR ($\kappa=3.0$)",
}

plot_eval_results_both(eval_result_dir, graphs2eval_results,
        xlabel="Step",
        ylabel=r"$\ell_1$ Distance",
        save_file="Different_kappa",
        graph2label=graph2label,
    )


# plot_eval_results(eval_result_dir, graphs2eval_results,
#         eval_result_type="weighted_dist",
#         xlabel="Step",
#         ylabel="Distance",
#         title="Weighted",
#         save_file="kappa_weighted_dist",
#         graph2label=graph2label,
#     )

# plot_eval_results(eval_result_dir, graphs2eval_results,
#         eval_result_type="unweighted_dist",
#         xlabel="Step",
#         ylabel="Distance",
#         title="Unweighted",
#         save_file="kappa_unweighted_dist",
#         graph2label=graph2label,
#     )

# L
ckpts = list(range(1000, 15000 + 1, 1000))
graphs2eval_results = {
    "STAR": [
        "graph=STAR_n=20_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_training_arguments=training_arguments_1_trial=1",
        "graph=STAR_n=20_training_arguments=training_arguments_1_trial=2",
        "graph=STAR_n=20_training_arguments=training_arguments_1_trial=3",
        "graph=STAR_n=20_training_arguments=training_arguments_1_trial=4",
    ],
    "X": [
        "graph=X_n=20_training_arguments=training_arguments_1_trial=0",
        "graph=X_n=20_training_arguments=training_arguments_1_trial=1",
        "graph=X_n=20_training_arguments=training_arguments_1_trial=2",
        "graph=X_n=20_training_arguments=training_arguments_1_trial=3",
        "graph=X_n=20_training_arguments=training_arguments_1_trial=4",
    ],
    "CHAIN": [
        "graph=CHAIN_n=20_training_arguments=training_arguments_1_trial=0",
        "graph=CHAIN_n=20_training_arguments=training_arguments_1_trial=1",
        "graph=CHAIN_n=20_training_arguments=training_arguments_1_trial=2",
        "graph=CHAIN_n=20_training_arguments=training_arguments_1_trial=3",
        "graph=CHAIN_n=20_training_arguments=training_arguments_1_trial=4",
    ],
}

graph2label = {
    "CHAIN": r"CHAIN ($L=18$)",
    "X": r"X ($L=9$)",
    "STAR": r"STAR ($L=1$)",
}

plot_eval_results_both(eval_result_dir, graphs2eval_results,
        xlabel="Step",
        ylabel=r"$\ell_1$ Distance",
        save_file="Different_L",
        graph2label=graph2label,
    )

# plot_eval_results(eval_result_dir, graphs2eval_results,
#         eval_result_type="weighted_dist",
#         xlabel="Step",
#         ylabel="Distance",
#         title="Weighted",
#         save_file="L_weighted_dist",
#         graph2label=graph2label,
#     )

# plot_eval_results(eval_result_dir, graphs2eval_results,
#         eval_result_type="unweighted_dist",
#         xlabel="Step",
#         ylabel="Distance",
#         title="Unweighted",
#         save_file="L_unweighted_dist",
#         graph2label=graph2label,
#     )



# n
ckpts = list(range(1000, 15000 + 1, 1000))
graphs2eval_results = {
    "n=10": [
        "graph=STAR_n=10_kappa=1.0_0_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=10_kappa=1.0_1_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=10_kappa=1.0_2_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=10_kappa=1.0_3_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=10_kappa=1.0_4_training_arguments=training_arguments_1_trial=0",
    ],
    "n=20": [
        "graph=STAR_n=20_kappa=1.0_0_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=1.0_1_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=1.0_2_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=1.0_3_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=20_kappa=1.0_4_training_arguments=training_arguments_1_trial=0",
    ],
    "n=30": [
        "graph=STAR_n=30_kappa=1.0_0_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=30_kappa=1.0_1_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=30_kappa=1.0_2_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=30_kappa=1.0_3_training_arguments=training_arguments_1_trial=0",
        "graph=STAR_n=30_kappa=1.0_4_training_arguments=training_arguments_1_trial=0",
    ],
}

graph2label = {
    "n=10": r"STAR ($n = 10$)",
    "n=20": r"STAR ($n = 20$)",
    "n=30": r"STAR ($n = 30$)",
}

plot_eval_results_both(eval_result_dir, graphs2eval_results,
        xlabel="Step",
        ylabel=r"$\ell_1$ Distance",
        save_file="Different_n",
        graph2label=graph2label,
    )

# plot_eval_results(eval_result_dir, graphs2eval_results,
#         eval_result_type="weighted_dist",
#         xlabel="Step",
#         ylabel="Distance",
#         title="Weighted",
#         save_file="n_weighted_dist",
#         graph2label=graph2label,
#     )

# plot_eval_results(eval_result_dir, graphs2eval_results,
#         eval_result_type="unweighted_dist",
#         xlabel="Step",
#         ylabel="Distance",
#         title="Unweighted",
#         save_file="n_unweighted_dist",
#         graph2label=graph2label,
#     )