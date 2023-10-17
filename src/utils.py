import argparse
import matplotlib.pyplot as plt
from models.GAT import GAT
from models.MessagePassing import MessagePassingGNN
from models.GraphSAGE import GraphSAGE
from models.GINE import GINE
from models.GAT_hetero import HeteroGAT
import os
import pandapower.plotting as ppl
import pandas as pd
import pandapower as pp
import pickle
import numpy as np
import random
import string
import torch as th
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils.convert import from_networkx
import tqdm
import math
import networkx as nx
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

def get_arguments():
    parser = argparse.ArgumentParser(prog="GNN script",
                                     description="Run a GNN to solve an inductive power system problem (power flow only for now)")
    
    parser.add_argument("gnn", choices=["GAT", "MessagePassing", "GraphSAGE", "GINE", "HeteroGAT"], default="GAT")
    # if file is moved in another directory level relative to the root (currently in root/src), this needs to be changed
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--train", default=root_directory + "/Data/train")
    parser.add_argument("--val", default=root_directory + "/Data/val")
    parser.add_argument("--test", default=root_directory + "/Data/test")
    parser.add_argument("-s", "--save_model", action="store_true", default=True)
    parser.add_argument("-m", "--model_name", default=''.join([random.choice(string.ascii_letters + string.digits) for _ in range(8)]))
    parser.add_argument("-p", "--plot", action="store_true", default=False)
    parser.add_argument("-o", "--optimizer", default="Adam")
    parser.add_argument("-c", "--criterion", default="MSELoss")
    parser.add_argument("-b", "--batch_size", default=16)
    parser.add_argument("-n", "--n_epochs", default=100)
    parser.add_argument("-l", "--learning_rate", default=1e-4)
    parser.add_argument("-w", "--weight_decay", default=0.05)
    parser.add_argument("--n_hidden_gnn", default=2, type=int)
    parser.add_argument("--gnn_hidden_dim", default=32, type=int)
    parser.add_argument("--n_hidden_lin", default=2, type=int)
    parser.add_argument("--lin_hidden_dim", default=8, type=int)
    parser.add_argument("--patience", default=40)
    parser.add_argument("--plot_node_error", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)

    args = parser.parse_args()
    return args

def load_data(train_dir, val_dir, test_dir):
    try:
        train = read_from_pkl(f"{train_dir}/pickled.pkl")
        val = read_from_pkl(f"{val_dir}/pickled.pkl")
        test = read_from_pkl(f"{test_dir}/pickled.pkl")
        print("Data Loaded from pkl files")
    except:
        print("Data not found, loading from json files...")
        print("Training Data...")
        train = load_data_helper(train_dir)
        print("Validation Data...")
        val = load_data_helper(val_dir)
        print("Testing Data...")
        test = load_data_helper(test_dir)
        print("hetero data can be loaded whooho")
        # quit()

        # save data to pkl
        write_to_pkl(train, f"{train_dir}/pickled.pkl")
        write_to_pkl(val, f"{val_dir}/pickled.pkl")
        write_to_pkl(test, f"{test_dir}/pickled.pkl")

        print("Data Loaded and saved to pkl files")

    return train, val, test

def load_data_helper(dir):
    graph_path = f"{dir}/x"
    sol_path = f"{dir}/y"
    graph_paths = sorted(os.listdir(graph_path))
    sol_paths = sorted(os.listdir(sol_path))
    data = []

    for i, g in tqdm.tqdm(enumerate(graph_paths)):
        graph = pp.from_json(f"{graph_path}/{g}")
        y_bus = pd.read_csv(f"{sol_path}/{sol_paths[i * 3]}", index_col=0)
        y_gen = pd.read_csv(f"{sol_path}/{sol_paths[i * 3 + 1]}", index_col=0)
        y_line = pd.read_csv(f"{sol_path}/{sol_paths[i * 3 + 2]}", index_col=0)

        instance = create_hetero_data_instance(graph, y_bus, y_gen, y_line)
        # Debug code
        # node_types, edge_types = instance.metadata()
        # print(node_types)
        # print(edge_types)
        # visualize_hetero(instance)
        data.append(instance)

    return data

def normalize_data(train, val, test, standard_normalizaton=True):
    # train, val and test are lists of torch_geometric.data.Data objects
    # create a tensor for x, y and edge_attr for all data (train, val, test)
    combined_x = th.cat([data.x for data in train + val + test], dim=0)
    combined_y = th.cat([data.y for data in train + val + test], dim=0)
    combined_edge_attr = th.cat([data.edge_attr for data in train + val + test], dim=0)

    epsilon = 1e-7  # to avoid division by zero

    # Standard normalization between -1 and 1
    if standard_normalizaton:

        # compute mean and std for all columns
        mean_x = th.mean(combined_x, dim=0)
        std_x = th.std(combined_x, dim=0)

        mean_y = th.mean(combined_y, dim=0) 
        std_y = th.std(combined_y, dim=0)

        mean_edge_attr = th.mean(combined_edge_attr, dim=0) 
        std_edge_attr = th.std(combined_edge_attr, dim=0)

        # normalize data
        for data in train + val + test:
            data.x = (data.x - mean_x) / (std_x + epsilon)
            data.y = (data.y - mean_y) / (std_y + epsilon)
            data.edge_attr = (data.edge_attr - mean_edge_attr) / (std_edge_attr + epsilon)
    
    else: # Use min max normalization to normalize data between 0 and 1 
        # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
        
        # find min value and max for all columns
        # x: vn_kv, p_mw_gen, vm_pu, p_mw_load, q_mvar
        min_x = th.min(combined_x, dim=0).values # tensor([     0.6000,   -681.7000,      0.0000,      0.0000,   -171.5000])
        max_x = th.max(combined_x, dim=0).values # tensor([  500.0000, 56834.0000,     1.1550, 57718.0000, 13936.0000])

        # y: p_mw, q_mvar, va_degree, vm_pu
        min_y = th.min(combined_y, dim=0).values # tensor([-11652.4385,  -5527.3564,   -156.9993,      0.0579])
        max_y = th.max(combined_y, dim=0).values # tensor([ 5844.1426,  1208.3413,   160.0282,     1.9177])

        # edge_attr: r_ohm_per_km, x_ohm_per_km, c_nf_per_km, g_us_per_km, max_i_ka, parallel, df, length_km
        min_edge_attr = th.min(combined_edge_attr, dim=0).values # tensor([  -296.9000,      0.0306,      0.0000,      0.0000,      0.0684,   1.0000,      1.0000,      1.0000])
        max_edge_attr = th.max(combined_edge_attr, dim=0).values # tensor([ 1152.5000,  1866.5001,  4859.9951,     0.0000, 99999.0000,     1.0000,   1.0000,     1.0000])

        # normalize data
        for data in train + val + test:
            data.x = (data.x - min_x) / (max_x - min_x + epsilon)
            data.y = (data.y - min_y) / (max_y - min_y + epsilon)
            data.edge_attr = (data.edge_attr - min_edge_attr) / (max_edge_attr - min_edge_attr + epsilon)

    return train, val, test

def create_hetero_data_instance(graph, y_bus, xxx, y_line):
    # Debug code
    # ppl.simple_plot(graph, plot_loads=True, plot_gens=True, trafo_color="r", switch_color="g") 
    # print(f"\nNumber of nodes: {graph.bus.shape[0]}")
    # print(f"Number of edges: {graph.line.shape[0]}")
    # print(f"Number of transformers: {graph.trafo.shape[0]}")
    # # print n of external grids, loads and generators
    # print(f"Number of external grids: {graph.ext_grid.shape[0]}")
    # print(f"Number of loads: {graph.load.shape[0]}")
    # print(f"Number of generators: {graph.gen.shape[0]}")

    # Get relevant values from gens, loads, and external grids TODO static generators
    gen = graph.gen[['bus', 'p_mw', 'vm_pu']]
    gen.rename(columns={'p_mw': 'p_mw_gen'}, inplace=True)
    gen['gen'] = 1
    gen.set_index('bus', inplace=True)

    load = graph.load[['bus', 'p_mw', 'q_mvar']]
    load.rename(columns={'p_mw': 'p_mw_load'}, inplace=True)
    load['load'] = 1
    load.set_index('bus', inplace=True)

    ext = graph.ext_grid[['bus', 'vm_pu', 'va_degree']]
    ext.rename(columns={'vm_pu': 'vm_pu_ext'}, inplace=True)
    ext['ext'] = 1
    ext.set_index('bus', inplace=True)
    
    # Merge to one dataframe
    node_feat = graph.bus[['vn_kv']]
    node_feat = node_feat.merge(gen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(load, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(ext, left_index=True, right_index=True, how='outer')

    # fill missing feature values with 0
    node_feat.fillna(0.0, inplace=True)
    node_feat['vm_pu'] = node_feat['vm_pu'] + node_feat['vm_pu_ext']
    node_feat['p_mw'] = node_feat['p_mw_load'] - node_feat['p_mw_gen']

    # remove duplicate columns/indices
    node_feat = node_feat[~node_feat.index.duplicated(keep='first')]
    node_feat['none'] = ((node_feat['gen'] == 0) & (node_feat['ext'] == 0) & (node_feat['load'] == 0)).astype(float)
    node_feat['load'] = node_feat['load'] + node_feat['none']
    node_feat['load_gen'] = ((node_feat['load'] == 1) & (node_feat['gen'] == 1)).astype(float)
    node_feat['load'] = ((node_feat['load'] == 1) & (node_feat['load_gen'] == 0) & (node_feat['ext'] == 0)).astype(float)
    node_feat['gen'] = ((node_feat['gen'] == 1) & (node_feat['load_gen'] == 0)).astype(float)

    # Select relevant columns
    node_feat = node_feat[['load', 'gen', 'load_gen', 'ext', 'p_mw', 'q_mvar', 'vm_pu', 'va_degree']]
    # Organize by type, but keep original indexing
    node_feat = pd.concat([
        node_feat[node_feat['load'] == 1],
        node_feat[node_feat['gen'] == 1],
        node_feat[node_feat['load_gen'] == 1],
        node_feat[node_feat['ext'] == 1]
        ], ignore_index=False)
    
    #Create index mapping to apply to target values and edges
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_feat.index)}
    node_feat = node_feat.reset_index(drop=True)

    # Extract relevant target data for each node type
    target = y_bus
    target.index = target.index.map(index_mapping)
    target.sort_index(inplace=True)

    node_types = ['load', 'gen', 'load_gen', 'ext']
    #Feature maps to get relevant x and y values
    feature_map = {
        'load': ['p_mw', 'q_mvar'],
        'gen': ['p_mw', 'vm_pu'],
        'load_gen': ['p_mw', 'q_mvar', 'vm_pu'],
        'ext': ['vm_pu', 'va_degree']
    }
    y_map = {
        'load': ['va_degree', 'vm_pu'],
        'gen': ['va_degree'],
        'load_gen': ['va_degree'],
        'ext': None
    }
    
    # Get edges, reindex and make bidirectional
    edge_index = graph.line[['from_bus', 'to_bus']].reset_index(drop=True)
    edge_index['from_bus'] = edge_index['from_bus'].map(index_mapping)
    edge_index['to_bus'] = edge_index['to_bus'].map(index_mapping)
    swapped = deepcopy(edge_index)
    swapped[['from_bus', 'to_bus']] = edge_index[['to_bus', 'from_bus']]
    bidirectional_edge_index = pd.concat([edge_index, swapped], axis=0)

    # Get edge attributes, reset index to match edge_index, and make bidirectional
    edge_attr = graph.line[['r_ohm_per_km',
                           'x_ohm_per_km',
                           'c_nf_per_km',
                           'g_us_per_km',
                           'max_i_ka',
                           'parallel',
                           'df',
                           'length_km']].reset_index(drop=True)
    
    bidirectional_edge_attr = pd.concat([edge_attr, edge_attr], axis=0)
    

    # Get edges for transformers, reindex, and make bidirectional
    edge_index_trafo = graph.trafo[['lv_bus', 'hv_bus']].reset_index(drop=True)
    edge_index_trafo['lv_bus'] = edge_index_trafo['lv_bus'].map(index_mapping)
    edge_index_trafo['hv_bus'] = edge_index_trafo['hv_bus'].map(index_mapping)
    swapped_trafo = deepcopy(edge_index_trafo)
    swapped_trafo[['lv_bus', 'hv_bus']] = edge_index_trafo[['hv_bus', 'lv_bus']]
    bidirectional_edge_index_trafo = pd.concat([edge_index_trafo, swapped_trafo], axis=0)
    
    
    # create a edge attribute dataframe for the transformers
    edge_attr_trafo = pd.DataFrame(columns=['one', 'two', 'three']) # what are the computed features called?
    edge_attr_list = []
    for trafo in graph.trafo.itertuples():
        # Calculate transformer attributes
        vkr_pu = float(trafo.vkr_percent / (trafo.sn_mva / (trafo.vn_lv_kv * math.sqrt(3))))
        p_pu = float(math.sqrt((trafo.vk_percent ** 2) - (trafo.vkr_percent) ** 2) / (trafo.sn_mva / (trafo.vn_lv_kv * math.sqrt(3))))
        q_pu = float(1.0)
        edge_attr_list.append([vkr_pu, p_pu, q_pu])
    
    edge_attr_trafo = pd.DataFrame(edge_attr_list, columns=['one', 'two', 'three'])

    # make the dataframe bidirectional
    bidirectional_edge_attr_trafo = pd.concat([edge_attr_trafo, edge_attr_trafo], axis=0)
        
    data = HeteroData()

    #Add features for each node type
    for node_type in node_types:
        mask = node_feat[node_type] == 1

        sub_df = node_feat[mask]
        features = feature_map[node_type]
        x = th.tensor(sub_df[features].values, dtype=th.float)
        data[node_type].x = x
        
        y_features = y_map[node_type]
        if y_features is not None:
            sub_df_y = target[mask]
            y = th.tensor(sub_df_y[y_features].values, dtype=th.float)
            data[node_type].y = y 


    # Add connecitons as nodes with edge attributes
    # data['connects'].edge_index = th.tensor(bidirectional_edge_index.values, dtype=th.long).t().contiguous()
    # data['connects'].edge_attributes = th.tensor(bidirectional_edge_attr.values, dtype=th.float)
    # iterate through each the nodes of each node type, create a dictionary that maps each node index to an increasing number from 0 to n_nodes of that tpye
    # then use that dictionary to map the edge_index and edge_attr to the new indices

    gen_dict = {old_idx: new_idx for new_idx, old_idx in enumerate(node_feat[node_feat['gen'] == 1].index)}
    load_dict = {old_idx: new_idx for new_idx, old_idx in enumerate(node_feat[node_feat['load'] == 1].index)}
    load_gen_dict = {old_idx: new_idx for new_idx, old_idx in enumerate(node_feat[node_feat['load_gen'] == 1].index)}
    ext_dict = {old_idx: new_idx for new_idx, old_idx in enumerate(node_feat[node_feat['ext'] == 1].index)}
    
    # print total entries in each dictionary
    # print("---")
    # print("total entries in each dictionary: ", len(gen_dict) + len(load_dict)+ len(load_gen_dict)+ len(ext_dict))
    # print("gen_dict ", gen_dict)
    # print("load_dict ", load_dict)
    # print("load_gen_dict ", load_gen_dict)
    # print("ext_dict ", ext_dict)
    # make a union of all the dictionaries
    new_dict = {}
    new_dict.update(load_dict)
    new_dict.update(gen_dict)
    new_dict.update(load_gen_dict)
    new_dict.update(ext_dict)
    # print("total entries in union dictionary:", len(new_dict))
    # print("new_dict", new_dict)
    # print(f"total number of nodes: {len(node_feat)}")
    # print("---")

    def map_indices(dataframe, index_mapping):
        # return a new dataframe where each column is mapped to the new indices
        new_df = pd.DataFrame(columns=dataframe.columns)
        for col in dataframe.columns:
            new_df[col] = dataframe[col].map(index_mapping)

        return new_df

    connection_types = [('connects', 'from_bus', 'to_bus', edge_index, edge_attr),
                         ('transformer', 'lv_bus', 'hv_bus', edge_index_trafo, edge_attr_trafo)]
    key_len = len(node_types)
    for (con_type, l, h, bei, bei_attr) in connection_types:
        # Same class connections
        for node_type in node_types:
            from_node_type = bei[l].isin(node_feat[node_feat[node_type] == 1].index)
            to_node_type = bei[h].isin(node_feat[node_feat[node_type] == 1].index)
            bidirectional_edge_index_node_type = bei[from_node_type & to_node_type]
            bidirectional_edge_attr_node_type = bei_attr.loc[bidirectional_edge_index_node_type.index]
            if bidirectional_edge_index_node_type.shape[0] > 0:
                # print(bidirectional_edge_index_node_type)
                # print(map_indices(bidirectional_edge_index_node_type, new_dict))
                data[node_type, con_type, node_type].edge_index = th.tensor(map_indices(bidirectional_edge_index_node_type, new_dict).values, dtype=th.long).t().contiguous()
                data[node_type, con_type, node_type].edge_attr = th.tensor(bidirectional_edge_attr_node_type.values, dtype=th.float)
            else:
                data[node_type, con_type, node_type].edge_index = th.tensor([], dtype=th.long).t().contiguous()
                data[node_type, con_type, node_type].edge_attr = th.tensor([], dtype=th.float)


        # Different class connections
        for i in range(key_len):
            node_type_a = node_types[i]
            for j in range(i + 1, key_len):
                node_type_b = node_types[j]
                from_bus_a = bei[l].isin(node_feat[node_feat[node_type_a] == 1].index)
                to_bus_b = bei[h].isin(node_feat[node_feat[node_type_b] == 1].index)
                from_bus_b = bei[l].isin(node_feat[node_feat[node_type_b] == 1].index)
                to_bus_a = bei[h].isin(node_feat[node_feat[node_type_a] == 1].index)
                bidirectional_edge_index_a_to_b = bei[from_bus_a & to_bus_b]
                bidirectional_edge_index_b_to_a = bei[from_bus_b & to_bus_a]
                bidirectional_edge_index_a_b_to_b = pd.concat([bidirectional_edge_index_a_to_b, bidirectional_edge_index_b_to_a], axis=0)
                mask = bidirectional_edge_index_a_b_to_b[l] > bidirectional_edge_index_a_b_to_b[h]
                bidirectional_edge_index_a_b_to_b.loc[mask, [l, h]] = bidirectional_edge_index_a_b_to_b.loc[mask, [h, l]].values
                bidirectional_edge_attr_a_b = bei_attr.loc[bidirectional_edge_index_a_b_to_b.index]
                if bidirectional_edge_index_a_b_to_b.shape[0] > 0:
                    data[node_type_a, con_type, node_type_b].edge_index = th.tensor(map_indices(bidirectional_edge_index_a_b_to_b, new_dict).values, dtype=th.long).t().contiguous()
                    data[node_type_a, con_type, node_type_b].edge_attr = th.tensor(bidirectional_edge_attr_a_b.values, dtype=th.float)
                else:
                    data[node_type_a, con_type, node_type_b].edge_index = th.tensor([], dtype=th.long).t().contiguous()
                    data[node_type_a, con_type, node_type_b].edge_attr = th.tensor([], dtype=th.float)
    # print(data)
    # ppl.simple_plot(graph)
    # visualize_hetero(data)
    # print("EDGES")
    # print(edge_index)
    # print(edge_index_trafo)
    # print("gen_dict ", gen_dict)
    # print("load_dict ", load_dict)
    # print("load_gen_dict ", load_gen_dict)
    # print("ext_dict ", ext_dict)
    # print("new_dict", new_dict)

    # for x in data.edge_types:
    #     l = len(data[x].edge_index)
    #     if l > 0:
    #         print(x)
    #         print(data[x].edge_index)
    # quit()
    return data

def visualize_hetero(hetero):
    G = nx.Graph()
    color_map = []
    total_nodes = 0
    
    # Adding nodes and setting colors
    for node_type in hetero.node_types:
        num_nodes = hetero[node_type].num_nodes
        G.add_nodes_from(range(total_nodes, total_nodes + num_nodes), node_type=node_type)
        total_nodes += num_nodes
        
        if node_type == 'load':
            color_map.extend(['red'] * num_nodes)
        elif node_type == 'gen':
            color_map.extend(['green'] * num_nodes)
        elif node_type == 'load_gen':
            color_map.extend(['blue'] * num_nodes)
        elif node_type == 'ext':
            color_map.extend(['gray'] * num_nodes)
        else:
            color_map.extend(['yellow'] * num_nodes)

    # Compute position of nodes
    pos = nx.spring_layout(G)
    
    # Draw the nodes with labels
    labels = {}
    for node_type, subgraph in hetero.items():
        for node_idx in subgraph.data.batch_num_nodes:
            labels[node_idx] = f"{node_type}-{node_idx}"
    # add legend for node types
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_weight="bold")

    # Adding edges for different edge types
    for edge_type in hetero.edge_types:
        # print(edge_type)
        edge_indices = hetero[edge_type].edge_index.t().numpy()

        if edge_type == ('load', 'connects', 'load'):
            edge_color = 'red'
            edge_label = 'load-load',
            edge_style = 'solid'
        elif edge_type == ('gen', 'connects', 'gen'):
            edge_color = 'green'
            edge_label = 'gen-gen'
            edge_style = 'solid'
        elif edge_type == ('load_gen', 'connects', 'load_gen'):
            edge_color = 'blue'
            edge_label = 'load_gen-load_gen'
            edge_style = 'solid'
        elif edge_type == ('ext', 'connects', 'ext'):
            edge_color = 'gray'
            edge_label = 'ext-ext'
            edge_style = 'solid'
        elif edge_type == ('load', 'connects', 'gen'):
            edge_color = 'orange'
            edge_label = 'load-gen'
            edge_style = 'dashed'
        elif edge_type == ('load', 'connects', 'ext'):
            edge_color = 'orange'
            edge_label = 'load-ext'
            edge_style = 'dashed'
        elif edge_type == ('gen', 'connects', 'ext'):
            edge_color = 'orange'
            edge_label = 'gen-ext'
            edge_style = 'dashed'
        elif edge_type == ('load', 'transformer', 'gen'):
            edge_color = 'red'
            edge_label = 'load-transf-gen'
            edge_style = 'dotted'
        elif edge_type == ('load', 'transformer', 'ext'):
            edge_color = 'red'
            edge_label = 'load-transf-ext'
            edge_style = 'dotted'
        elif edge_type == ('gen', 'transformer', 'ext'):
            edge_color = 'red'
            edge_label = 'gen-transf-ext'
            edge_style = 'dotted'
        elif edge_type == ('load', 'transformer', 'load'):
            edge_color = 'red'
            edge_label = 'load-transf-load'
            edge_style = 'dotted'
        elif edge_type == ('gen', 'transformer', 'gen'):
            edge_color = 'red'
            edge_label = 'gen-transf-gen'
            edge_style = 'dotted'
        elif edge_type == ('load_gen', 'transformer', 'load_gen'):
            edge_color = 'red'
            edge_label = 'load_gen-transf-load_gen'
            edge_style = 'dotted'
        elif edge_type == ('load_gen', 'connects', 'ext'):
            edge_color = 'orange'
            edge_label = 'ext-load_gen'
            edge_style = 'solid'
        elif edge_type == ('load', 'connects', 'load_gen'):
            edge_color = 'purple'
            edge_label = 'load-load_gen'
            edge_style = 'dashed'
        elif edge_type == ('load', 'transformer', 'load_gen'):
            edge_color = 'green'
            edge_label = 'load-transf-load_gen'
            edge_style = 'dashed'
        else:
            edge_color = 'black'
            edge_label = 'other'
            edge_style = 'dotted'

               
        nx.draw_networkx_edges(G, pos, edgelist=edge_indices, edge_color=edge_color, style=edge_style, label=edge_label, width=1)

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=200, node_color=color_map, font_size=15, font_weight="bold")
    
    # Add a legend at the top left corner of the plot
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    # Show the plot
    plt.show()

# return a torch_geometric.data.Data object for each instance
def create_data_instance(graph, y_bus, y_gen, y_line):
    g = ppl.create_nxgraph(graph, include_trafos=True)
    # []
    # https://pandapower.readthedocs.io/en/latest/elements/gen.html
    gen = graph.gen[['bus', 'p_mw', 'vm_pu']]
    gen.rename(columns={'p_mw': 'p_mw_gen'}, inplace=True)
    gen['gen'] = 1
    gen.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/load.html
    load = graph.load[['bus', 'p_mw', 'q_mvar']]
    load.rename(columns={'p_mw': 'p_mw_load'}, inplace=True)
    load.set_index('bus', inplace=True)

    ext = graph.ext_grid[['bus', 'vm_pu', 'va_degree']]
    ext.rename(columns={'vm_pu': 'vm_pu_ext'}, inplace=True)
    ext['ext'] = 1
    ext.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/bus.html
    node_feat = graph.bus[['vn_kv']]

    # make sure all nodes (bus, gen, load) have the same number of features (namely the union of all features)
    node_feat = node_feat.merge(gen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(load, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(ext, left_index=True, right_index=True, how='outer')

    # fill missing feature values with 0
    node_feat.fillna(0.0, inplace=True)
    node_feat['vm_pu'] = node_feat['vm_pu'] + node_feat['vm_pu_ext']
    node_feat['p_mw'] = node_feat['p_mw_load'] - node_feat['p_mw_gen']

    del node_feat['vm_pu_ext']
    del node_feat['p_mw_gen']
    del node_feat['p_mw_load']
    del node_feat['vn_kv']

    # remove duplicate columns/indices
    node_feat = node_feat[~node_feat.index.duplicated(keep='first')]
    node_feat['load'] = (node_feat['gen'] == 0) & (node_feat['ext'] == 0)
    node_feat['load'] = node_feat['load'].astype(float)
    node_feat = node_feat[['load', 'gen', 'ext', 'p_mw', 'q_mvar', 'vm_pu', 'va_degree']]
    # zero_check = node_feat[(node_feat['load'] == 0) & (node_feat['gen'] == 0) & (node_feat['ext'] == 0) & (node_feat['is_none'] == 0)]
    # load_ext_check = node_feat[(node_feat['load'] == 1) & (node_feat['ext'] == 1)]

    # if not zero_check.empty:
    #     print("zero check failed")
    #     print(node_feat)
    #     print("zero check results")
    #     print(zero_check)
    #     quit()
    
    # if not load_ext_check.empty:
    #     print("load ext check failed")
    #     print(node_feat)
    #     print("load ext check results")
    #     print(zero_check)
    #     quit()
    

    for node in node_feat.itertuples():
        # set each node features
        g.nodes[node.Index]['x'] = [float(node.load), 
                                    float(node.gen), 
                                    float(node.ext), 
                                    float(node.p_mw), 
                                    float(node.q_mvar), 
                                    float(node.vm_pu),
                                    float(node.va_degree)]
        
        g.nodes[node.Index]['y'] = [float(y_bus['va_degree'][node.Index]),
                                    float(y_bus['vm_pu'][node.Index])]
        
        # g.nodes[node.Index]['y'] = [float(y_bus['p_mw'][node.Index]),
        #                             float(y_bus['q_mvar'][node.Index]),
        #                             float(y_bus['vm_pu'][node.Index]),
        #                             float(y_bus['va_degree'][node.Index])]
        # set each node label by type
        # if node.load:
        #     g.nodes[node.Index]['y'] = [float(y_bus['va_degree'][node.Index]),
        #                                 float(y_bus['vm_pu'][node.Index])]
        #     # g.nodes[node.Index]['reals'] = [float(y_bus['p_mw'][node.Index]),
        #     #                                 float(y_bus['q_mvar'][node.Index])]
        # elif node.gen:
        #     g.nodes[node.Index]['y'] = [float(y_bus['q_mvar'][node.Index]),
        #                                 float(y_bus['va_degree'][node.Index])]
        #     # g.nodes[node.Index]['reals'] = [float(y_bus['p_mw'][node.Index]),
        #     #                                 float(y_bus['vm_pu'][node.Index])]
        # elif node.ext:
        #     g.nodes[node.Index]['y'] = [float(y_bus['p_mw'][node.Index]),
        #                                 float(y_bus['q_mvar'][node.Index])]
        #     # g.nodes[node.Index]['reals'] = [float(y_bus['vm_pu'][node.Index]),
        #     #                                 float(y_bus['va_degree'][node.Index])]
        # else:
        #     g.nodes[node.Index]['y'] = [float(y_bus['va_degree'][node.Index]),
        #                                 float(y_bus['vm_pu'][node.Index])]
        #     # g.nodes[node.Index]['reals'] = [float(y_bus['p_mw'][node.Index]),
        #     #                                 float(y_bus['q_mvar'][node.Index])]
        

    first = True
    for edges in graph.line.itertuples():
        if first:
            common_edge = edges
            first = False
        g.edges[edges.from_bus, edges.to_bus, ('line', edges.Index)]['edge_attr'] = [float(edges.r_ohm_per_km),
                                                                                     float(edges.x_ohm_per_km),
                                                                                     float(edges.length_km)]
                                                                                    #  float(edges.c_nf_per_km),
                                                                                    #  float(edges.g_us_per_km),
                                                                                    #  float(edges.max_i_ka),
                                                                                    #  float(edges.parallel),
                                                                                    #  float(edges.df),

    # print(common_edge)
    for trafos in graph.trafo.itertuples():
        g.edges[trafos.lv_bus, trafos.hv_bus, ('trafo', trafos.Index)]['edge_attr'] = [float(trafos.vkr_percent / (trafos.sn_mva / (trafos.vn_lv_kv * math.sqrt(3)))),
                                                                                       float(math.sqrt((trafos.vk_percent ** 2) - (trafos.vkr_percent) ** 2)) / (trafos.sn_mva / (trafos.vn_lv_kv * math.sqrt(3))),
                                                                                       1.0]
                                                                                    #  float(common_edge.c_nf_per_km),
                                                                                    #  float(common_edge.g_us_per_km),
                                                                                    #  float(common_edge.max_i_ka),
                                                                                    #  float(common_edge.parallel),
                                                                                    #  float(common_edge.df),
    return from_networkx(g)


def get_gnn(gnn_name):
    if gnn_name == "GAT":
        return GAT
    
    if gnn_name == "MessagePassing":
        return MessagePassingGNN
    
    if gnn_name == "GraphSAGE":
        return GraphSAGE
    
    if gnn_name == "GINE":
        return GINE
    
    if gnn_name == "HeteroGAT":
        return HeteroGAT

def get_optim(optim_name):
    if optim_name == "Adam":
        return th.optim.Adam
    

def get_criterion(criterion_name):
    if criterion_name == "MSELoss":
        return nn.MSELoss()
    if criterion_name == "L1Loss":
        return nn.L1Loss()
    
def save_model(model, model_name):
    state = {
        'model': model, # save the model object with some of its parameters
        'state_dict': model.state_dict(),
    }
    # timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    model_name = model_name + "-" + model.class_name # + "_" + str(timestamp)
    # if file is moved in another directory level relative to the root (currently in root/src), this needs to be changed
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_directory = root_directory + "/trained_models"
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    th.save(model.state_dict(), f"{save_directory}/{model_name}.pt")
    

def load_model(gnn_type, path, data, arguments):
    input_dim = data[0].x.shape[1]
    edge_attr_dim = data[0].edge_attr.shape[1] 
    output_dim = data[0].y.shape[1]
    gnn_class = get_gnn(gnn_type)
    model = gnn_class(input_dim, 
                      output_dim, 
                      edge_attr_dim,
                      arguments.n_hidden_gnn, 
                      arguments.gnn_hidden_dim, 
                      arguments.n_hidden_lin, 
                      arguments.lin_hidden_dim)
    model.load_state_dict(th.load(path))
    return model


def write_to_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def read_from_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data