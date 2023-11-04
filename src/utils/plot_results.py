import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils_plot import plot_percent_curve

# # FULL graph

# n_hidden_conv=5,
# hidden_conv_dim=24,
# n_heads=1,
# n_hidden_lin=1,
# hidden_lin_dim=38,
# dropout_rate=0.3,
# conv_type='GINE', # GAT or GATv2 or SAGE or GINE
# jumping_knowledge='mean', # max or lstm or mean, None to disable
# hetero_aggr='sum', # sum or mean or max or mul

# RND neighbor

# n_hidden_conv=2,
# hidden_conv_dim=24,
# n_heads=1,
# n_hidden_lin=1,
# hidden_lin_dim=38,
# dropout_rate=0.2,
# conv_type='GINE', # GAT or GATv2 or SAGE or GINE
# jumping_knowledge='lstm', # max or lstm or mean, None to disable
# hetero_aggr='sum', # sum or mean or max or mul

# RND walk

# n_hidden_conv=5,
# hidden_conv_dim=24,
# n_heads=1,
# n_hidden_lin=1,
# hidden_lin_dim=38,
# dropout_rate=0.3,
# conv_type='GINE', # GAT or GATv2 or SAGE or GINE
# jumping_knowledge='mean', # max or lstm or mean, None to disable
# hetero_aggr='sum', # sum or mean or max or mul

# BFS

# n_hidden_conv=2,
# hidden_conv_dim=24,
# n_heads=1,
# n_hidden_lin=1,
# hidden_lin_dim=38,
# dropout_rate=0.3,
# conv_type='GINE', 
# jumping_knowledge='lstm', 
# hetero_aggr='sum',


COLS = ['load_vm_pu','load_va_deg','gen_va_deg','load_gen_va_deg', 'va_degree', 'q_mvar']

# Edge Features
print("Edge Features")
d_edge_feat = {
    'GINE': './results/edge_features/GINE-edge-attrs.csv',
    'GINE no feat': './results/edge_features/GINE_no_edge_attrs.csv',
    'GraphSAGE': './results/edge_features/SAGE-edge-attrs.csv',
    'GraphSAGE no feat': './results/edge_features/SAGE-no_edge_attrs.csv',
}

for c in COLS:
    plot_percent_curve(d_edge_feat, c)

# N-1
print("N - 1")
d_minus = {
    'N - 0': './Data/results/minus/physics_0_results.csv',
    'N - 1': './Data/results/minus/physics_1_results.csv',
    'N - 2': './Data/results/minus/physics_2_results.csv',
    'N - 3': './Data/results/minus/physics_3_results.csv',
}

for c in COLS:
    plot_percent_curve(d_minus, c)

# Subgraphing Methods
print("Subgraphing Methods")
d_subgraphing = {
    'BFS': './Data/results/subgraphing/bfs.csv',
    'Random Walk': './Data/results/subgraphing/rnd_walk.csv',
    'Random Neighbor': './Data/results/subgraphing/rnd_neighbor.csv',

}

for c in COLS:
    plot_percent_curve(d_subgraphing, c)

# Supervised / Unsupervised
print("Supervised / Unsupervised")
supervised = {
    'Supervised': './Data/results/supervised/supervised.csv',
    'Unsupervised': './Data/results/supervised/unsupervised.csv',
}

for c in COLS:
    plot_percent_curve(supervised, c)