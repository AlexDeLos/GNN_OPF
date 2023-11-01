import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def plot_within(data, names, title):
    for i, d in enumerate(data):
        plt.plot(range(1, 101), d, label=names[i])
    plt.title(title)
    plt.legend()
    plt.xlabel("Error Threshold in %")
    plt.ylabel("Percent within error threshold")
    plt.show()

bfs = pd.read_csv('./Data/results/subgraphing/bfs.csv')
rnd_neighbor = pd.read_csv('./Data/results/subgraphing/rnd_neighbor.csv')
rnd_walk = pd.read_csv('./Data/results/subgraphing/rnd_walk.csv')

names = ['BFS', 'Random Neighbor', 'Random Walk']
titles  = ['Load Voltage Magnitude', 'Load Voltage Angle', 'Load & Generator Voltage Angle', 'Generator Voltage Angle']

for i, col in enumerate(bfs.columns.values.tolist()):
    bfs_data = bfs[col].to_numpy()
    neighbor_data = rnd_neighbor[col].to_numpy()
    walk_data = rnd_walk[col].to_numpy()
    plot_within([bfs_data, neighbor_data, walk_data], names, titles[i])

# # Load

# load_vm_pu_bfs = bfs['load_vm_pu'].to_numpy()
# load_vm_pu_rnd_neighbor = rnd_neighbor['load_vm_pu'].to_numpy()
# load_vm_pu_rnd_walk = rnd_walk['load_vm_pu'].to_numpy()

# load_va_degree_bfs = bfs['load_va_deg'].to_numpy()
# load_va_degree_rnd_neighbor = rnd_neighbor['load_va_deg'].to_numpy()
# load_va_degree_rnd_walk = rnd_walk['load_va_deg'].to_numpy()

# # Load / Generator

# load_gen_va_degree_bfs = bfs['load_gen_va_deg'].to_numpy()
# load_gen_va_degree_rnd_neighbor = rnd_neighbor['load_gen_va_deg'].to_numpy()
# load_gen_va_degree_rnd_walk = rnd_walk['load_gen_va_deg'].to_numpy()

# # Gen

# gen_va_degree_bfs = bfs['gen_va_deg'].to_numpy()
# gen_va_degree_rnd_neighbor = rnd_neighbor['gen_va_deg'].to_numpy()
# gen_va_degree_rnd_walk = rnd_walk['gen_va_deg'].to_numpy()
