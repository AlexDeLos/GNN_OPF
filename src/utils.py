import argparse
import math
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
import random
import string
import torch as th
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils.convert import from_networkx
import tqdm
from utils_hetero import create_hetero_data_instance


def get_arguments():
    parser = argparse.ArgumentParser(prog="GNN script",
                                     description="Run a GNN to solve an inductive power system problem (power flow only for now)")
    
    # Important: prefix all heterogeneous GNNs names with "Hetero"
    parser.add_argument("gnn", choices=["GAT", "MessagePassing", "GraphSAGE", "GINE", "HeteroGAT"], default="GAT")
    # if file is moved in another directory level relative to the root (currently in root/src), this needs to be changed
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--train", default=root_directory + "/Data/train")
    parser.add_argument("--val", default=root_directory + "/Data/val")
    parser.add_argument("--test", default=root_directory + "/Data/test")
    parser.add_argument("-s", "--save_model", action="store_true", default=True)
    parser.add_argument("-m", "--model_name", default=''.join([random.choice(string.ascii_letters + string.digits) for _ in range(8)]))
    parser.add_argument("-p", "--plot", action="store_true", default=True)
    parser.add_argument("-o", "--optimizer", default="Adam")
    parser.add_argument("-c", "--criterion", default="MSELoss")
    parser.add_argument("-b", "--batch_size", default=128)
    parser.add_argument("-n", "--n_epochs", default=100)
    parser.add_argument("-l", "--learning_rate", default=1e-2)
    parser.add_argument("-w", "--weight_decay", default=1e-5)
    parser.add_argument("--n_hidden_gnn", default=1, type=int)
    parser.add_argument("--gnn_hidden_dim", default=16, type=int)
    parser.add_argument("--n_hidden_lin", default=0, type=int)
    parser.add_argument("--lin_hidden_dim", default=32, type=int)
    parser.add_argument("--patience", default=15)
    parser.add_argument("--plot_node_error", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--physics", action="store_true", default=False)
    parser.add_argument("--no_linear", action="store_true", default=False)
    parser.add_argument("--value_mode", choices=['all', 'missing', 'voltage'], default='all')

    args = parser.parse_args()
    return args


def load_data(train_dir, val_dir, test_dir, gnn_type, load_physics=False, missing=False, volt=False):
    try:
        train = read_from_pkl(f"{train_dir}/pickled.pkl")
        val = read_from_pkl(f"{val_dir}/pickled.pkl")
        test = read_from_pkl(f"{test_dir}/pickled.pkl")
        print("Data Loaded from pkl files")
    except:
        print("Data not found, loading from json files...")
        print("Training Data...")

        train = load_data_helper(train_dir, gnn_type, physics_data=load_physics, missing=missing, volt=volt)
        print("Validation Data...")
        val = load_data_helper(val_dir, gnn_type, physics_data=load_physics, missing=missing, volt=volt)
        print("Testing Data...")
        test = load_data_helper(test_dir, gnn_type, physics_data=load_physics, missing=missing, volt=volt)

        # save data to pkl
        write_to_pkl(train, f"{train_dir}/pickled.pkl")
        write_to_pkl(val, f"{val_dir}/pickled.pkl")
        write_to_pkl(test, f"{test_dir}/pickled.pkl")

        print("Data Loaded and saved to pkl files")

    return train, val, test

  
def load_data_helper(dir, gnn_type, physics_data=False, missing=False, volt=False):
    graph_path = f"{dir}/x"
    sol_path = f"{dir}/y"
    graph_paths = sorted(os.listdir(graph_path))
    sol_paths = sorted(os.listdir(sol_path))
    data = []

    for i, g in tqdm.tqdm(enumerate(graph_paths)):
        graph = pp.from_json(f"{graph_path}/{g}")
        y_bus = pd.read_csv(f"{sol_path}/{sol_paths[i * 3]}", index_col=0)
        # y_gen = pd.read_csv(f"{sol_path}/{sol_paths[i * 3 + 1]}", index_col=0)
        # y_line = pd.read_csv(f"{sol_path}/{sol_paths[i * 3 + 2]}", index_col=0)

        if gnn_type[:6] != "Hetero":
            instance = create_physics_data_instance(graph, y_bus, missing, volt)
        else:
            instance = create_hetero_data_instance(graph, y_bus)
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

    else:  # Use min max normalization to normalize data between 0 and 1
        # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)

        # find min value and max for all columns
        # x: vn_kv, p_mw_gen, vm_pu, p_mw_load, q_mvar
        min_x = th.min(combined_x,
                       dim=0).values  # tensor([     0.6000,   -681.7000,      0.0000,      0.0000,   -171.5000])
        max_x = th.max(combined_x, dim=0).values  # tensor([  500.0000, 56834.0000,     1.1550, 57718.0000, 13936.0000])

        # y: p_mw, q_mvar, va_degree, vm_pu
        min_y = th.min(combined_y, dim=0).values  # tensor([-11652.4385,  -5527.3564,   -156.9993,      0.0579])
        max_y = th.max(combined_y, dim=0).values  # tensor([ 5844.1426,  1208.3413,   160.0282,     1.9177])

        # edge_attr: r_ohm_per_km, x_ohm_per_km, c_nf_per_km, g_us_per_km, max_i_ka, parallel, df, length_km
        min_edge_attr = th.min(combined_edge_attr,
                               dim=0).values  # tensor([  -296.9000,      0.0306,      0.0000,      0.0000,      0.0684,   1.0000,      1.0000,      1.0000])
        max_edge_attr = th.max(combined_edge_attr,
                               dim=0).values  # tensor([ 1152.5000,  1866.5001,  4859.9951,     0.0000, 99999.0000,     1.0000,   1.0000,     1.0000])

        # normalize data
        for data in train + val + test:
            data.x = (data.x - min_x) / (max_x - min_x + epsilon)
            data.y = (data.y - min_y) / (max_y - min_y + epsilon)
            data.edge_attr = (data.edge_attr - min_edge_attr) / (max_edge_attr - min_edge_attr + epsilon)

    return train, val, test


# return a torch_geometric.data.Data object for each instance
def create_data_instance(graph, y_bus, missing, volt):
    g = ppl.create_nxgraph(graph, include_trafos=True)
    # https://pandapower.readthedocs.io/en/latest/elements/gen.html
    gen = graph.gen[['bus', 'p_mw', 'vm_pu']]
    gen.rename(columns={'p_mw': 'p_mw_gen'}, inplace=True)
    gen['is_gen'] = 1
    gen.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/sgen.html
    # Note: multiple static generators can be attached to 1 bus!
    sgen = graph.sgen[['bus', 'p_mw', 'q_mvar']]
    sgen.rename(columns={'p_mw': 'p_mw_sgen'}, inplace=True)
    sgen.rename(columns={'q_mvar': 'q_mvar_sgen'}, inplace=True)
    sgen = sgen.groupby('bus')[['p_mw_sgen', 'q_mvar_sgen']].sum()  # Already resets index
    sgen['is_sgen'] = 1

    # https://pandapower.readthedocs.io/en/latest/elements/load.html
    load = graph.load[['bus', 'p_mw', 'q_mvar']]
    load.rename(columns={'p_mw': 'p_mw_load'}, inplace=True)
    load.rename(columns={'q_mvar': 'q_mvar_load'}, inplace=True)
    load['is_load'] = 1
    load.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/ext_grid.html
    ext = graph.ext_grid[['bus', 'vm_pu', 'va_degree']]
    ext.rename(columns={'vm_pu': 'vm_pu_ext'}, inplace=True)
    ext['is_ext'] = 1
    ext.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/shunt.html
    shunt = graph.shunt[['bus', 'q_mvar']]
    shunt.rename(columns={'q_mvar': 'q_mvar_shunt'}, inplace=True)
    shunt.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/bus.html
    node_feat = graph.bus[['vn_kv']]

    # make sure all nodes (bus, gen, load) have the same number of features (namely the union of all features)
    node_feat = node_feat.merge(gen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(sgen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(load, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(ext, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(shunt, left_index=True, right_index=True, how='outer')

    # fill missing feature values with 0
    node_feat.fillna(0.0, inplace=True)
    node_feat['vm_pu'] = node_feat['vm_pu'] + node_feat['vm_pu_ext']
    node_feat['p_mw'] = node_feat['p_mw_load'] - node_feat['p_mw_gen'] - node_feat['p_mw_sgen']
    node_feat['q_mvar'] = node_feat['q_mvar_load'] + node_feat['q_mvar_shunt'] - node_feat['q_mvar_sgen']

    # static generators are modeled as loads in PandaPower
    node_feat['is_load'] = (node_feat['is_sgen'] != 0) | (node_feat['is_load'] != 0)

    del node_feat['vm_pu_ext']
    del node_feat['p_mw_gen']
    del node_feat['p_mw_sgen']
    del node_feat['p_mw_load']
    del node_feat['q_mvar_load']
    del node_feat['q_mvar_sgen']
    del node_feat['q_mvar_shunt']
    del node_feat['is_sgen']

    # remove duplicate columns/indices
    node_feat = node_feat[~node_feat.index.duplicated(keep='first')]
    node_feat['is_none'] = (node_feat['is_gen'] == 0) & (node_feat['is_load'] == 0) & (node_feat['is_ext'] == 0)
    node_feat['is_none'] = node_feat['is_none'].astype(float)
    node_feat = node_feat[['is_load', 'is_gen', 'is_ext', 'is_none', 'p_mw', 'q_mvar', 'va_degree', 'vm_pu']]
    zero_check = node_feat[(node_feat['is_load'] == 0) & (node_feat['is_gen'] == 0) & (node_feat['is_ext'] == 0) & (
            node_feat['is_none'] == 0)]

    if not zero_check.empty:
        print("zero check failed")
        print(node_feat)
        print("zero check results")
        print(zero_check)
        quit()

    for node in node_feat.itertuples():
        # set each node features
        g.nodes[node.Index]['x'] = [float(node.is_load),
                                    float(node.is_gen),
                                    float(node.is_ext),
                                    float(node.is_none),
                                    float(node.p_mw),
                                    float(node.q_mvar),
                                    float(node.va_degree),
                                    float(node.vm_pu)]

        if missing:
            if node.is_load or node.is_none:
                g.nodes[node.Index]['y'] = [float(y_bus['vm_pu'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
            if node.is_gen and not node.is_load:
                g.nodes[node.Index]['y'] = [float(y_bus['q_mvar'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
            if node.is_ext:
                g.nodes[node.Index]['y'] = [float(y_bus['vm_pu'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
        elif volt:
            g.nodes[node.Index]['y'] = [float(y_bus['vm_pu'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
             
        else:
            g.nodes[node.Index]['y'] = [float(y_bus['p_mw'][node.Index]),
                                        float(y_bus['q_mvar'][node.Index]),
                                        float(y_bus['vm_pu'][node.Index]),
                                        float(y_bus['va_degree'][node.Index])]

    for edges in graph.line.itertuples():
        g.edges[edges.from_bus, edges.to_bus, ('line', edges.Index)]['edge_attr'] = [float(edges.r_ohm_per_km),
                                                                                     float(edges.x_ohm_per_km),
                                                                                     float(edges.length_km)]

    for trafos in graph.trafo.itertuples():
        g.edges[trafos.lv_bus, trafos.hv_bus, ('trafo', trafos.Index)]['edge_attr'] = [float(trafos.vkr_percent / (trafos.sn_mva / (trafos.vn_lv_kv * math.sqrt(3)))),
                                                                                       float(math.sqrt((trafos.vk_percent ** 2) - (trafos.vkr_percent) ** 2)) / (trafos.sn_mva / (trafos.vn_lv_kv * math.sqrt(3))),
                                                                                       1.0]

    return from_networkx(g)


# Create the data needed for the physics loss
# return a torch_geometric.data.Data object for each instance
def create_physics_data_instance(graph, y_bus, missing, volt):
    # Convert PandaPower graph to NetworkX graph and set it to be directed (for directed transformer edges further down)
    g = ppl.create_nxgraph(graph, include_trafos=True)
    g = g.to_directed()

    # https://pandapower.readthedocs.io/en/latest/elements/gen.html
    gen = graph.gen[['bus', 'p_mw', 'vm_pu']]
    gen.rename(columns={'p_mw': 'p_mw_gen'}, inplace=True)
    gen['is_gen'] = 1
    gen.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/sgen.html
    # Note: multiple static generators can be attached to 1 bus!
    sgen = graph.sgen[['bus', 'p_mw', 'q_mvar']]
    sgen.rename(columns={'p_mw': 'p_mw_sgen'}, inplace=True)
    sgen.rename(columns={'q_mvar': 'q_mvar_sgen'}, inplace=True)
    sgen = sgen.groupby('bus')[['p_mw_sgen', 'q_mvar_sgen']].sum()  # Already resets index
    sgen['is_sgen'] = 1

    # https://pandapower.readthedocs.io/en/latest/elements/load.html
    load = graph.load[['bus', 'p_mw', 'q_mvar']]
    load.rename(columns={'p_mw': 'p_mw_load'}, inplace=True)
    load.rename(columns={'q_mvar': 'q_mvar_load'}, inplace=True)
    load['is_load'] = 1
    load.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/ext_grid.html
    ext = graph.ext_grid[['bus', 'vm_pu', 'va_degree']]
    ext.rename(columns={'vm_pu': 'vm_pu_ext'}, inplace=True)
    ext_degree = ext.loc[0, 'va_degree']
    if ext_degree != 30.0:
        print(ext_degree)
    ext['is_ext'] = 1
    ext.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/shunt.html
    shunt = graph.shunt[['bus', 'q_mvar', 'step']]
    shunt['b_pu_shunt'] = shunt['q_mvar'] * shunt['step'] / graph.sn_mva
    shunt.rename(columns={'q_mvar': 'q_mvar_shunt'}, inplace=True)
    del shunt['step']
    shunt.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/bus.html
    node_feat = graph.bus[['vn_kv']]

    # make sure all nodes (bus, gen, load) have the same number of features (namely the union of all features)
    node_feat = node_feat.merge(gen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(sgen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(load, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(ext, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(shunt, left_index=True, right_index=True, how='outer')

    # fill missing feature values with 0
    node_feat.fillna(0.0, inplace=True)
    node_feat['vm_pu'] = node_feat['vm_pu'] + node_feat['vm_pu_ext']
    node_feat['p_mw'] = node_feat['p_mw_load'] - node_feat['p_mw_gen'] - node_feat['p_mw_sgen']
    node_feat['q_mvar'] = node_feat['q_mvar_load'] + node_feat['q_mvar_shunt'] - node_feat['q_mvar_sgen']

    # static generators are modeled as loads in PandaPower
    node_feat['is_load'] = (node_feat['is_sgen'] != 0) | (node_feat['is_load'] != 0)

    del node_feat['vm_pu_ext']
    del node_feat['p_mw_gen']
    del node_feat['p_mw_sgen']
    del node_feat['p_mw_load']
    del node_feat['q_mvar_load']
    del node_feat['q_mvar_sgen']
    del node_feat['q_mvar_shunt']
    del node_feat['is_sgen']

    # remove duplicate columns/indices
    node_feat = node_feat[~node_feat.index.duplicated(keep='first')]
    node_feat['is_none'] = (node_feat['is_gen'] == 0) & (node_feat['is_load'] == 0) & (node_feat['is_ext'] == 0)
    node_feat['is_none'] = node_feat['is_none'].astype(float)
    node_feat = node_feat[['is_load', 'is_gen', 'is_ext', 'is_none', 'p_mw', 'q_mvar', 'va_degree', 'vm_pu', 'b_pu_shunt']]
    zero_check = node_feat[(node_feat['is_load'] == 0) & (node_feat['is_gen'] == 0) & (node_feat['is_ext'] == 0) & (
                node_feat['is_none'] == 0)]

    if not zero_check.empty:
        print("zero check failed")
        print(node_feat)
        print("zero check results")
        print(zero_check)
        quit()

    for node in node_feat.itertuples():
        # set each node features
        g.nodes[node.Index]['x'] = [float(node.is_load),
                                    float(node.is_gen),
                                    float(node.is_ext),
                                    float(node.is_none),
                                    float(node.p_mw / graph.sn_mva),
                                    float(node.q_mvar / graph.sn_mva),
                                    float(node.vm_pu),
                                    float(node.va_degree),
                                    float(node.b_pu_shunt)]

        if missing:
            if node.is_load or node.is_none:
                g.nodes[node.Index]['y'] = [float(y_bus['vm_pu'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
            if node.is_gen and not node.is_load:
                g.nodes[node.Index]['y'] = [float(y_bus['q_mvar'][node.Index] / graph.sn_mva),
                            float(y_bus['va_degree'][node.Index])]
            if node.is_ext:
                g.nodes[node.Index]['y'] = [float(y_bus['p_mw'][node.Index] / graph.sn_mva),
                            float(y_bus['q_mvar'][node.Index]) / graph.sn_mva]
        elif volt:
            g.nodes[node.Index]['y'] = [float(y_bus['vm_pu'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
             
        else:
            g.nodes[node.Index]['y'] = [float(y_bus['p_mw'][node.Index] / graph.sn_mva),
                                    float(y_bus['q_mvar'][node.Index] / graph.sn_mva),
                                    float(y_bus['vm_pu'][node.Index]),
                                    float(y_bus['va_degree'][node.Index])]

    for edges in graph.line.itertuples():
        # Calculate line admittance from impedance and convert to per-unit system
        r_tot = float(edges.r_ohm_per_km) * float(edges.length_km)
        x_tot = float(edges.x_ohm_per_km) * float(edges.length_km)
        conductance_line, susceptance_line = impedance_to_admittance(r_tot, x_tot, graph.bus['vn_kv'][edges.from_bus], graph.sn_mva)
        g.edges[edges.from_bus, edges.to_bus, ('line', edges.Index)]['edge_attr'] = [float(conductance_line), float(susceptance_line)]
        g.edges[edges.to_bus, edges.from_bus, ('line', edges.Index)]['edge_attr'] = [float(conductance_line), float(susceptance_line)]

    for trafos in graph.trafo.itertuples():
        # First calculate values from low to high voltage bus
        r_tot = 0.0
        x_tot = trafos.vk_percent * (trafos.vn_lv_kv ** 2) / trafos.sn_mva
        conductance, susceptance = impedance_to_admittance(r_tot, x_tot, trafos.vn_lv_kv, graph.sn_mva)
        g.edges[trafos.lv_bus, trafos.hv_bus, ('trafo', trafos.Index)]['edge_attr'] = [float(conductance), float(susceptance)]

        # Now high to low voltage bus values
        r_tot = 0.0
        x_tot = trafos.vk_percent * (trafos.vn_hv_kv ** 2) / trafos.sn_mva
        conductance, susceptance = impedance_to_admittance(r_tot, x_tot, trafos.vn_hv_kv, graph.sn_mva)
        g.edges[trafos.hv_bus, trafos.lv_bus, ('trafo', trafos.Index)]['edge_attr'] = [float(conductance), float(susceptance)]

    return from_networkx(g)


def impedance_to_admittance(r_ohm, x_ohm, base_volt, rated_power, per_unit_conversion=True):
    if per_unit_conversion:
        z_base = base_volt ** 2 / rated_power  # Z_base used to convert impedance to per-unit
        r_tot = r_ohm / z_base  # Convert to per unit metrics before converting to admittance
        x_tot = x_ohm / z_base
    else:
        r_tot = r_ohm
        x_tot = x_ohm
    denom = r_tot ** 2 + x_tot ** 2
    conductance = r_tot / denom
    susceptance = -x_tot / denom
    return conductance, susceptance


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
    if optim_name == "Adadelta":
        return th.optim.Adadelta
    if optim_name == "Adagrad":
        return th.optim.Adagrad
    if optim_name == "AdamW":
        return th.optim.AdamW
    if optim_name == "SparseAdam":
        return th.optim.SparseAdam
    if optim_name == "Adamax":
        return th.optim.Adamax
    if optim_name == "ASGD":
        return th.optim.ASGD
    if optim_name == "LBFGS":
        return th.optim.LBFGS
    if optim_name == "NAdam":
        return th.optim.NAdam
    if optim_name == "RAdam":
        return th.optim.RAdam
    if optim_name == "RMSProp":
        return th.optim.RMSProp
    if optim_name == "Rprop":
        return th.optim.Rprop
    if optim_name == "SGD":
        return th.optim.SGD


def get_criterion(criterion_name):
    if criterion_name == "MSELoss":
        return nn.MSELoss()
    if criterion_name == "L1Loss":
        return nn.L1Loss()
    if criterion_name == "Huber":
        return nn.HuberLoss()


def save_model(model, model_name):
    state = {
        'model': model,  # save the model object with some of its parameters
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
                      arguments.lin_hidden_dim,
                      no_lin=arguments.no_linear)
    print(model)
    model.load_state_dict(th.load(path))
    return model

def load_model_hetero(gnn_type, path, data, arguments):
    output_dims = {node_type: data[0].y_dict[node_type].shape[1] for node_type in data[0].y_dict.keys()}
    gnn_class = get_gnn(gnn_type)
    gnn = gnn_class(output_dim_dict=output_dims, 
                    edge_types=data[0].edge_index_dict.keys(),
                    n_hidden_conv=arguments.n_hidden_gnn,
                    hidden_conv_dim = arguments.gnn_hidden_dim,
                    n_hidden_lin=arguments.n_hidden_lin,
                    hidden_lin_dim = arguments.lin_hidden_dim
                    )
    gnn.load_state_dict(th.load(path))
    return gnn

def write_to_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_from_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data