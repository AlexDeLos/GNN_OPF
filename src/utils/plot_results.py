import numpy as np
import matplotlib.pyplot as plt


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

# Load

load_vm_pu_bfs = np.array([])
 
load_vm_pu_rnd_walk = np.array([])
load_vm_pu_rnd_neighbor = np.array([])

load_va_degree_bfs = np.array([])
load_va_degree_rnd_walk = np.array([])
load_va_degree_rnd_neighbor = np.array([])

# Load / Generator

load_gen_va_degree_bfs = np.array([])
load_gen_va_degree_rnd_walk = np.array([])
load_gen_va_degree_rnd_neighbor = np.array([])

# Gen

gen_va_degree_bfs = np.array([])
gen_va_degree_rnd_walk = np.array([])
gen_va_degree_rnd_neighbor = np.array([])
