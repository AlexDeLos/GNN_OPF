from copy import deepcopy
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import networkx as nx
import sys
import torch as th
from torch_geometric.data import HeteroData
import torch_geometric.utils as pyg_util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_physics import impedance_to_admittance
import warnings
# suppress the UserWarning
warnings.filterwarnings('ignore', '.*Boolean Series key will be reindexed.*')


def create_hetero_data_instance(graph, y_bus, physics_data=False):
    # Get relevant values from gens, loads, and external grids
    gen = graph.gen[['bus', 'p_mw', 'vm_pu']]
    gen.rename(columns={'p_mw': 'p_mw_gen'}, inplace=True)
    gen['gen'] = 1
    gen.set_index('bus', inplace=True)
    
    # https://pandapower.readthedocs.io/en/latest/elements/sgen.html
    # Note: multiple static generators can be attached to 1 bus!
    sgen = graph.sgen[['bus', 'p_mw', 'q_mvar']]
    sgen.rename(columns={'p_mw': 'p_mw_sgen'}, inplace=True)
    sgen.rename(columns={'q_mvar': 'q_mvar_sgen'}, inplace=True)
    sgen = sgen.groupby('bus')[['p_mw_sgen', 'q_mvar_sgen']].sum()  # Already resets index
    sgen['sgen'] = 1

    load = graph.load[['bus', 'p_mw', 'q_mvar']]
    load.rename(columns={'p_mw': 'p_mw_load'}, inplace=True)
    load.rename(columns={'q_mvar': 'q_mvar_load'}, inplace=True)
    load['load'] = 1
    load.set_index('bus', inplace=True)

    ext = graph.ext_grid[['bus', 'vm_pu', 'va_degree']]
    ext.rename(columns={'vm_pu': 'vm_pu_ext'}, inplace=True)
    ext_degree = ext.loc[0, 'va_degree']
    # if ext_degree != 30.0:
    #     print('ext_degree')
    #     print(ext_degree)
    ext['ext'] = 1
    ext.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/shunt.html
    shunt = graph.shunt[['bus', 'q_mvar', 'step']]
    shunt['b_pu_shunt'] = shunt['q_mvar'] * shunt['step'] / graph.sn_mva
    shunt.rename(columns={'q_mvar': 'q_mvar_shunt'}, inplace=True)
    del shunt['step']
    shunt.set_index('bus', inplace=True)
    
    # Merge to one dataframe
    node_feat = graph.bus[['vn_kv']]

    node_feat = node_feat.merge(gen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(sgen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(load, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(ext, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(shunt, left_index=True, right_index=True, how='outer')

    # fill missing feature values with 0
    node_feat.fillna(0.0, inplace=True)
    node_feat['vm_pu'] = node_feat['vm_pu'] + node_feat['vm_pu_ext']
    node_feat['p_mw'] = node_feat['p_mw_load'] - node_feat['p_mw_gen'] - node_feat['p_mw_sgen']
    node_feat['p_mw'] = node_feat['p_mw'] / graph.sn_mva
    node_feat['q_mvar'] = node_feat['q_mvar_load'] + node_feat['q_mvar_shunt'] - node_feat['q_mvar_sgen']
    node_feat['q_mvar'] = node_feat['q_mvar'] / graph.sn_mva

    # static generators are modeled as loads in PandaPower
    node_feat['load'] = (node_feat['sgen'] != 0) | (node_feat['load'] != 0)
    
    del node_feat['vm_pu_ext']
    del node_feat['p_mw_gen']
    del node_feat['p_mw_sgen']
    del node_feat['p_mw_load']
    del node_feat['q_mvar_load']
    del node_feat['q_mvar_sgen']
    del node_feat['q_mvar_shunt']
    del node_feat['sgen']

    node_feat['none'] = ((node_feat['gen'] == 0) & (node_feat['ext'] == 0) & (node_feat['load'] == 0)).astype(float)
    node_feat['load'] = node_feat['load'] + node_feat['none']
    node_feat['load_gen'] = ((node_feat['load'] == 1) & (node_feat['gen'] == 1)).astype(float)
    node_feat['load'] = ((node_feat['load'] == 1) & (node_feat['load_gen'] == 0) & (node_feat['ext'] == 0)).astype(float)
    node_feat['gen'] = ((node_feat['gen'] == 1) & (node_feat['load_gen'] == 0)).astype(float)

    # remove duplicate columns/indices
    node_feat = node_feat[~node_feat.index.duplicated(keep='first')]
    node_feat = node_feat[['load', 'gen', 'load_gen', 'ext', 'p_mw', 'q_mvar', 'va_degree', 'vm_pu']]
    zero_check = node_feat[(node_feat['load'] == 0) & (node_feat['gen'] == 0) & (node_feat['ext'] == 0) & (node_feat['load_gen'] == 0)]

    if not zero_check.empty:
        print("zero check failed")
        print(node_feat)
        print("zero check results")
        print(zero_check)
        quit()

    # Organize by type, but keep original indexing
    node_feat = pd.concat([
        node_feat[node_feat['load'] == 1],
        node_feat[node_feat['gen'] == 1],
        node_feat[node_feat['load_gen'] == 1],
        node_feat[node_feat['ext'] == 1]
        ], ignore_index=False)
    
    # Create index mapping to apply to target values and edges
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_feat.index)}
    node_feat = node_feat.reset_index(drop=True)

    # Extract relevant target data for each node type
    target = y_bus
    # Convert power values to per unit here as well
    target['p_mw'] = target['p_mw'] / graph.sn_mva
    target['q_mvar'] = target['q_mvar'] / graph.sn_mva
    target.index = target.index.map(index_mapping)
    target.sort_index(inplace=True)

    node_types = ['load', 'gen', 'load_gen', 'ext']
    # Feature maps to get relevant x and y values
    # Using the physics loss requires having all 4 (p_mw, q_mvar, vm_pu,va_degree) features per node (combined as input or output)
    if physics_data:
        feature_map = {
            'load': ['p_mw', 'q_mvar'],
            'gen': ['p_mw', 'vm_pu'],
            'load_gen': ['p_mw', 'q_mvar', 'vm_pu'],
            'ext': ['vm_pu', 'va_degree']
        }
        y_map = {
            'load': ['va_degree', 'vm_pu'],
            'gen': ['q_mvar', 'va_degree'],
            'load_gen': ['q_mvar', 'va_degree'],
            'ext': ['p_mw', 'q_mvar']
        }
    else:
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

    edge_attr_lines_list = []
    for edges in graph.line.itertuples():
        # Calculate line admittance from impedance and convert to per-unit system
        r_tot = float(edges.r_ohm_per_km) * float(edges.length_km)
        x_tot = float(edges.x_ohm_per_km) * float(edges.length_km)
        conductance_line, susceptance_line = impedance_to_admittance(r_tot, x_tot, graph.bus['vn_kv'][edges.from_bus], graph.sn_mva)
        edge_attr_lines_list.append([conductance_line, susceptance_line])
    
    edge_attr = pd.DataFrame(edge_attr_lines_list, columns=['conductance', 'susceptance'])

    # Get edges for transformers, reindex, and make bidirectional
    edge_index_trafo = graph.trafo[['lv_bus', 'hv_bus']].reset_index(drop=True)
    edge_index_trafo['lv_bus'] = edge_index_trafo['lv_bus'].map(index_mapping)
    edge_index_trafo['hv_bus'] = edge_index_trafo['hv_bus'].map(index_mapping)
    
    # create a edge attribute dataframe for the transformers
    edge_attr_trafo_list = []
    for trafo in graph.trafo.itertuples():
        r_tot = 0.0
        x_tot = (trafo.vk_percent / 100) * (trafo.vn_lv_kv ** 2) / trafo.sn_mva
        conductance, susceptance = impedance_to_admittance(r_tot, x_tot, trafo.vn_lv_kv, graph.sn_mva)
        edge_attr_trafo_list.append([conductance, susceptance])
    
    edge_attr_trafo = pd.DataFrame(edge_attr_trafo_list, columns=['conductance', 'susceptance'])
        
    data = HeteroData()

    # Add features for each node type
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

    
    # iterate through each the nodes of each node type, create a dictionary that maps each node index to an increasing number from 0 to n_nodes of that tpye
    # then use that dictionary to map the edge_index and edge_attr to the new indices
    gen_dict = {old_idx: new_idx for new_idx, old_idx in enumerate(node_feat[node_feat['gen'] == 1].index)}
    load_dict = {old_idx: new_idx for new_idx, old_idx in enumerate(node_feat[node_feat['load'] == 1].index)}
    load_gen_dict = {old_idx: new_idx for new_idx, old_idx in enumerate(node_feat[node_feat['load_gen'] == 1].index)}
    ext_dict = {old_idx: new_idx for new_idx, old_idx in enumerate(node_feat[node_feat['ext'] == 1].index)}
    
    new_dict = {}
    new_dict.update(load_dict)
    new_dict.update(gen_dict)
    new_dict.update(load_gen_dict)
    new_dict.update(ext_dict)

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
                data[node_type, con_type, node_type].edge_index = th.tensor(map_indices(bidirectional_edge_index_node_type, new_dict).values, dtype=th.long).t().contiguous()
                data[node_type, con_type, node_type].edge_attr = th.tensor(bidirectional_edge_attr_node_type.values, dtype=th.float)

                # Add reverse direction edges as well for bidirectional edges instead of one directional
                reverse_edge_indices = th.stack((data[node_type, con_type, node_type].edge_index[1], data[node_type, con_type, node_type].edge_index[0]))
                data[node_type, con_type, node_type].edge_index = th.cat((data[node_type, con_type, node_type].edge_index, reverse_edge_indices), -1)
                data[node_type, con_type, node_type].edge_attr = th.cat((data[node_type, con_type, node_type].edge_attr, data[node_type, con_type, node_type].edge_attr))
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

        # add reverse edges (T.ToUndirected() didn't work so we do it manually)
        for i in range(key_len):
            node_type_a = node_types[i]
            for j in range(i + 1, key_len):
                node_type_b = node_types[j]
                data[node_type_b, con_type, node_type_a].edge_index = data[node_type_a, con_type, node_type_b].edge_index.flip(0)
                data[node_type_b, con_type, node_type_a].edge_attr = data[node_type_a, con_type, node_type_b].edge_attr

    # visualize_hetero(data)
    return data


def normalize_data_hetero(train, val, test, standard_normalization=True):
    combined_x_dict = {}
    combined_y_dict = {}
    combined_edge_attr_dict = {}

    for data in train + val + test:
        for key, value in data.x_dict.items():
            if key not in combined_x_dict:
                combined_x_dict[key] = value.clone()
            else:
                combined_x_dict[key] = th.cat([combined_x_dict[key], value], dim=0)

        for key, value in data.y_dict.items():
            if key not in combined_y_dict:
                combined_y_dict[key] = value.clone()
            else:
                combined_y_dict[key] = th.cat([combined_y_dict[key], value], dim=0)

        for key, value in data.edge_attr_dict.items():
            if key not in combined_edge_attr_dict:
                combined_edge_attr_dict[key] = value.clone()
            else:
                combined_edge_attr_dict[key] = th.cat([combined_edge_attr_dict[key], value], dim=0)

    epsilon = 1e-7  # to avoid division by zero

    # Standard normalization between -1 and 1
    if standard_normalization:

        # compute mean and std for all columns
        mean_x_dict = {}
        std_x_dict = {}
        mean_y_dict = {}
        std_y_dict = {}
        mean_edge_attr_dict = {}
        std_edge_attr_dict = {}

        for key, value in combined_x_dict.items():
            mean_x_dict[key] = th.mean(value, dim=0)
            std_x_dict[key] = th.std(value, dim=0)

        for key, value in combined_y_dict.items():
            mean_y_dict[key] = th.mean(value, dim=0)
            std_y_dict[key] = th.std(value, dim=0)

        for key, value in combined_edge_attr_dict.items():
            mean_edge_attr_dict[key] = th.mean(value, dim=0)
            std_edge_attr_dict[key] = th.std(value, dim=0)

        # normalize data
        for data in train + val + test:
            # one line for loop to replace full dictionary at once, replacing one key at a time doesnt work
            data.x_dict = {k: (v - mean_x_dict[k]) / (std_x_dict[k] + epsilon) for k, v in data.x_dict.items()}
            data.y_dict = {k: (v - mean_y_dict[k]) / (std_y_dict[k] + epsilon) for k, v in data.y_dict.items()}
            # v.numel() > 0 checks for empty tensors (e.g. zero entries for a specific edge type)
            data.edge_attr_dict = {k: ((v - mean_edge_attr_dict[k]) / (std_edge_attr_dict[k] + epsilon) if v.numel() > 0 else v) for k, v in data.edge_attr_dict.items()}

    else: # Use min max normalization to normalize data between 0 and 1 
        # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
        
        # find min value and max for all columns
        # x: vn_kv, p_mw_gen, vm_pu, p_mw_load, q_mvar
        min_x_dict = {}
        max_x_dict = {}
        min_y_dict = {}
        max_y_dict = {}
        min_edge_attr_dict = {}
        max_edge_attr_dict = {}

        for key, value in combined_x_dict.items():
            min_x_dict[key] = th.min(value, dim=0).values
            max_x_dict[key] = th.max(value, dim=0).values

        for key, value in combined_y_dict.items():
            min_y_dict[key] = th.min(value, dim=0).values
            max_y_dict[key] = th.max(value, dim=0).values

        for key, value in combined_edge_attr_dict.items():
            if value.numel() > 0:
                min_edge_attr_dict[key] = th.min(value, dim=0).values
                max_edge_attr_dict[key] = th.max(value, dim=0).values
            else:
                min_edge_attr_dict[key] = th.tensor([])
                max_edge_attr_dict[key] = th.tensor([])

        # normalize data
        for data in train + val + test:
            data.x_dict = {k: (v - min_x_dict[k]) / (max_x_dict[k] - min_x_dict[k] + epsilon) for k, v in data.x_dict.items()}
            data.y_dict = {k: (v - min_y_dict[k]) / (max_y_dict[k] - min_y_dict[k] + epsilon) for k, v in data.y_dict.items()}
            data.edge_attr_dict = {k: ((v - min_edge_attr_dict[k]) / (max_edge_attr_dict[k] - min_edge_attr_dict[k] + epsilon) if v.numel() > 0 else v) for k, v in data.edge_attr_dict.items()}
    return train, val, test


def physics_loss_hetero(data, output, log_loss=True, per_node_type=False, device='cpu'):
    # Take all edge type and compute -1 * feature 1 and -1 * feature 2 to get off-diagonal admittance matrix values
    conductance_dict = {}
    susceptance_dict = {}
    for edge_type in data.edge_attr_dict.keys():
        if data.edge_attr_dict[edge_type].numel() > 0:
            conductance_dict[edge_type] = -1.0 * data.edge_attr_dict[edge_type][:, 0]
            susceptance_dict[edge_type] = -1.0 * data.edge_attr_dict[edge_type][:, 1]

    feature_map = {
        'load': ['p_mw', 'q_mvar'],
        'gen': ['p_mw', 'vm_pu'],
        'load_gen': ['p_mw', 'q_mvar', 'vm_pu'],
        'ext': ['vm_pu', 'va_degree']
    }
    y_map = {
        'load': ['va_degree', 'vm_pu'],
        'gen': ['q_mvar', 'va_degree'],
        'load_gen': ['q_mvar', 'va_degree'],
        'ext': ['p_mw', 'q_mvar']
    }

    # Create tensors for each node type to contain the combined fixed inputs and predicted missing outputs
    # Also create empty tensors for each node type to create the calculated active and reactive values for each node
    combined_output = {}
    active_power = {}
    reactive_power = {}
    cols = ['p_mw', 'q_mvar', 'vm_pu', 'va_degree']
    for node_type in data.x_dict.keys():
        num_nodes = data.y_dict[node_type].shape[0]
        combined_output_node = th.zeros((num_nodes, 4), dtype=th.float, device=device)

        if node_type == 'load':
            # Add predicted vm_pu and va_degree
            combined_output_node[:, cols.index('vm_pu')] += output[node_type][:, y_map[node_type].index('vm_pu')]
            combined_output_node[:, cols.index('va_degree')] += output[node_type][:, y_map[node_type].index('va_degree')]
            # Add fixed p_mw and q_mvar
            combined_output_node[:, cols.index('p_mw')] += data.x_dict[node_type][:, feature_map[node_type].index('p_mw')]
            combined_output_node[:, cols.index('q_mvar')] += data.x_dict[node_type][:, feature_map[node_type].index('q_mvar')]

        elif node_type == 'gen' or node_type == 'load_gen':
            # Add predicted q_mvar and va_degree
            combined_output_node[:, cols.index('q_mvar')] += output[node_type][:, y_map[node_type].index('q_mvar')]
            combined_output_node[:, cols.index('va_degree')] += output[node_type][:, y_map[node_type].index('va_degree')]
            # Add fixed p_mw and vm_pu
            combined_output_node[:, cols.index('p_mw')] += data.x_dict[node_type][:, feature_map[node_type].index('p_mw')]
            combined_output_node[:, cols.index('vm_pu')] += data.x_dict[node_type][:, feature_map[node_type].index('vm_pu')]

        elif node_type == 'ext':
            # Add predicted p_mw and q_mvar
            combined_output_node[:, cols.index('p_mw')] += output[node_type][:, y_map[node_type].index('p_mw')]
            combined_output_node[:, cols.index('q_mvar')] += output[node_type][:, y_map[node_type].index('q_mvar')]
            # Add fixed vm_pu and va_degree
            combined_output_node[:, cols.index('vm_pu')] += data.x_dict[node_type][:, feature_map[node_type].index('vm_pu')]
            combined_output_node[:, cols.index('va_degree')] += data.x_dict[node_type][:, feature_map[node_type].index('va_degree')]

        # Add resulting combined fixed/predicted features tensor and create the empty power flow calculation tensors
        combined_output[node_type] = combined_output_node
        active_power[node_type] = th.zeros(num_nodes, dtype=th.float, device=device)
        reactive_power[node_type] = th.zeros(num_nodes, dtype=th.float, device=device)

    # Calculating active/reactive power flow of the 'from' nodes per edge type and adding them all up in the corresponding node_type tensors
    for edge_type in data.edge_attr_dict.keys():
        if data.edge_attr_dict[edge_type].numel() > 0:
            from_type = edge_type[0]
            to_type = edge_type[2]

            from_nodes = pyg_util.select(combined_output[from_type], data.edge_index_dict[edge_type][0], 0)
            to_nodes = pyg_util.select(combined_output[to_type], data.edge_index_dict[edge_type][1], 0)
            angle_diffs = (from_nodes[:, cols.index('va_degree')] - to_nodes[:, cols.index('va_degree')]) * math.pi / 180.0  # list of angle differences for all edges

            # calculate incoming/outgoing values based on the edges connected to each node and the node's + neighbour's values
            act_imb = th.abs(from_nodes[:, cols.index('vm_pu')]) * th.abs(to_nodes[:, cols.index('vm_pu')]) * (conductance_dict[edge_type] * th.cos(angle_diffs) + susceptance_dict[edge_type] * th.sin(angle_diffs))  # per edge power flow into/out of from_nodes
            rea_imb = th.abs(from_nodes[:, cols.index('vm_pu')]) * th.abs(to_nodes[:, cols.index('vm_pu')]) * (conductance_dict[edge_type] * th.sin(angle_diffs) - susceptance_dict[edge_type] * th.cos(angle_diffs))

            # aggregate all active and reactive powers for each node
            # not all node indices may appear in the edge_index_dict[edge_type], so need to keep track of which indices are there and were aggregated in order to add them to the correct node's power flow values
            num_nodes = active_power[from_type].shape[0]
            aggr_act_imb = pyg_util.scatter(act_imb, data.edge_index_dict[edge_type][0], dim_size=num_nodes)
            aggr_rea_imb = pyg_util.scatter(rea_imb, data.edge_index_dict[edge_type][0], dim_size=num_nodes)

            # add diagonal (self-admittance) elements of each node as well (for the ones that occur in this specific edge_type, others will be 0 due to the scatter result)
            aggr_act_imb += combined_output[from_type][:, cols.index('vm_pu')] * combined_output[from_type][:, cols.index('vm_pu')] * pyg_util.scatter(data.edge_attr_dict[edge_type][:, 0], data.edge_index_dict[edge_type][0], dim_size=num_nodes)
            aggr_rea_imb += combined_output[from_type][:, cols.index('vm_pu')] * combined_output[from_type][:, cols.index('vm_pu')] * (-1.0 * pyg_util.scatter(data.edge_attr_dict[edge_type][:, 1], data.edge_index_dict[edge_type][0], dim_size=num_nodes))

            # Add values to the power flow tensors for the respective node type
            active_power[from_type] += aggr_act_imb
            reactive_power[from_type] += aggr_rea_imb

    # Calculate power imbalance for each node type based on fixed/predicted p_mw and q_mvar and the calculated power flows
    active_imbalance = {}
    reactive_imbalance = {}
    for node_type in data.x_dict.keys():
        active_imbalance[node_type] = -1.0 * combined_output[node_type][:, cols.index('p_mw')] - active_power[node_type]
        reactive_imbalance[node_type] = -1.0 * combined_output[node_type][:, cols.index('q_mvar')] - reactive_power[node_type]

    # Use either sum of absolute imbalances or log of squared imbalances
    if log_loss:
        aggr_imbalances = th.tensor([0.0], device=device)
        for node_type in data.x_dict.keys():
            aggr_imbalances += th.sum(active_imbalance[node_type] * active_imbalance[node_type] + reactive_imbalance[node_type] * reactive_imbalance[node_type])
        tot_loss = th.log(1.0 + aggr_imbalances)
    elif not log_loss and per_node_type:
        tot_loss = {}
        for node_type in data.x_dict.keys():
            tot_loss[node_type] = th.sum(th.abs(active_imbalance[node_type]) + th.abs(reactive_imbalance[node_type]))
    else:
        tot_loss = th.tensor([0.0], device=device)
        for node_type in data.x_dict.keys():
            tot_loss += th.sum(th.abs(active_imbalance[node_type]) + th.abs(reactive_imbalance[node_type]))

    return tot_loss


def power_from_voltages_hetero(data, voltages, angles_are_radians=False):
    """
    Calculates the active and reactive power from the given voltage values (for hetero data).

    Args:
        data: power grid network (in hetero format)
        voltages: hetero voltage results (dict of node types with corresponding voltage tensors)
        angles_are_radians: angle values are given in radians or degrees

    Returns:
        p_mw and q_mvar of each node per node type
    """
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

    conductance_dict = {}
    susceptance_dict = {}
    for edge_type in data.edge_attr_dict.keys():
        if data.edge_attr_dict[edge_type].numel() > 0:
            conductance_dict[edge_type] = -1.0 * data.edge_attr_dict[edge_type][:, 0]
            susceptance_dict[edge_type] = -1.0 * data.edge_attr_dict[edge_type][:, 1]

    # Create tensors for each node type to contain the voltage values (depends on node type which is fixed/predicted)
    # Also create empty tensors for each node type to create the calculated active and reactive values for each node
    combined_output = {}
    active_power = {}
    reactive_power = {}
    cols = ['vm_pu', 'va_degree']
    node_types = list(data.x_dict.keys())
    node_types.append('ext')
    for node_type in node_types:
        num_nodes = data.x_dict[node_type].shape[0]
        combined_output_node = th.zeros((num_nodes, 2), dtype=th.float)

        if node_type == 'load':
            # Both vm_pu and va_degree are predicted for loads
            combined_output_node[:, cols.index('vm_pu')] += voltages[node_type][:, y_map[node_type].index('vm_pu')]
            combined_output_node[:, cols.index('va_degree')] += voltages[node_type][:, y_map[node_type].index('va_degree')]

        elif node_type == 'gen' or node_type == 'load_gen':
            # Add predicted va_degree
            combined_output_node[:, cols.index('va_degree')] += voltages[node_type][:, y_map[node_type].index('va_degree')]
            # Add fixed vm_pu
            combined_output_node[:, cols.index('vm_pu')] += data.x_dict[node_type][:, feature_map[node_type].index('vm_pu')]

        elif node_type == 'ext':
            # Add fixed vm_pu and va_degree
            combined_output_node[:, cols.index('vm_pu')] += data.x_dict[node_type][:, feature_map[node_type].index('vm_pu')]
            combined_output_node[:, cols.index('va_degree')] += data.x_dict[node_type][:, feature_map[node_type].index('va_degree')]

        # Add resulting combined fixed/predicted features tensor and create the empty power flow calculation tensors
        combined_output[node_type] = combined_output_node
        active_power[node_type] = th.zeros(num_nodes, dtype=th.float)
        reactive_power[node_type] = th.zeros(num_nodes, dtype=th.float)


    for edge_type in data.edge_attr_dict.keys():
        if data.edge_attr_dict[edge_type].numel() > 0:
            from_type = edge_type[0]
            to_type = edge_type[2]

            from_nodes = pyg_util.select(combined_output[from_type], data.edge_index_dict[edge_type][0], 0)
            to_nodes = pyg_util.select(combined_output[to_type], data.edge_index_dict[edge_type][1], 0)
            angle_diffs = (from_nodes[:, cols.index('va_degree')] - to_nodes[:, cols.index('va_degree')]) * math.pi / 180.0  # list of angle differences for all edges

            # calculate incoming/outgoing values based on the edges connected to each node and the node's + neighbour's values
            act_imb = th.abs(from_nodes[:, cols.index('vm_pu')]) * th.abs(to_nodes[:, cols.index('vm_pu')]) * (conductance_dict[edge_type] * th.cos(angle_diffs) + susceptance_dict[edge_type] * th.sin(angle_diffs))  # per edge power flow into/out of from_nodes
            rea_imb = th.abs(from_nodes[:, cols.index('vm_pu')]) * th.abs(to_nodes[:, cols.index('vm_pu')]) * (conductance_dict[edge_type] * th.sin(angle_diffs) - susceptance_dict[edge_type] * th.cos(angle_diffs))

            # aggregate all active and reactive powers for each node
            # not all node indices may appear in the edge_index_dict[edge_type], so need to keep track of which indices are there and were aggregated in order to add them to the correct node's power flow values
            num_nodes = active_power[from_type].shape[0]
            aggr_act_imb = pyg_util.scatter(act_imb, data.edge_index_dict[edge_type][0], dim_size=num_nodes)
            aggr_rea_imb = pyg_util.scatter(rea_imb, data.edge_index_dict[edge_type][0], dim_size=num_nodes)

            # add diagonal (self-admittance) elements of each node as well (for the ones that occur in this specific edge_type, others will be 0 due to the scatter result)
            aggr_act_imb += combined_output[from_type][:, cols.index('vm_pu')] * combined_output[from_type][:, cols.index('vm_pu')] * pyg_util.scatter(data.edge_attr_dict[edge_type][:, 0], data.edge_index_dict[edge_type][0], dim_size=num_nodes)
            aggr_rea_imb += combined_output[from_type][:, cols.index('vm_pu')] * combined_output[from_type][:, cols.index('vm_pu')] * (-1.0 * pyg_util.scatter(data.edge_attr_dict[edge_type][:, 1], data.edge_index_dict[edge_type][0], dim_size=num_nodes))

            # Add values to the power flow tensors for the respective node type
            active_power[from_type] += aggr_act_imb
            reactive_power[from_type] += aggr_rea_imb

    res = {}
    for node_type in data.x_dict.keys():
        active_power[node_type] = (-1.0 * active_power[node_type]).reshape(-1, 1)
        reactive_power[node_type] = (-1.0 * reactive_power[node_type]).reshape(-1, 1)
        res[node_type] = th.cat((active_power[node_type], reactive_power[node_type]), 1)

    return res


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