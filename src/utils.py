import argparse
import matplotlib.pyplot as plt
from models.GAT import GAT
from models.MessagePassing import MessagePassingGNN
from models.GraphSAGE import GraphSAGE
from models.GINE import GINE
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
from torch_geometric.utils.convert import from_networkx
import tqdm


def get_arguments():
    parser = argparse.ArgumentParser(prog="GNN script",
                                     description="Run a GNN to solve an inductive power system problem (power flow only for now)")
    
    parser.add_argument("gnn", choices=["GAT", "MessagePassing", "GraphSAGE", "GINE"], default="GAT")
    parser.add_argument("--train", default="./Data/train")
    parser.add_argument("--val", default="./Data/val")
    parser.add_argument("--test", default="./Data/test")
    parser.add_argument("-s", "--save_model", action="store_true", default=True)
    parser.add_argument("-m", "--model_name", default=''.join([random.choice(string.ascii_letters + string.digits) for _ in range(8)]))
    parser.add_argument("-p", "--plot", action="store_true", default=True)
    parser.add_argument("-o", "--optimizer", default="Adam")
    parser.add_argument("-c", "--criterion", default="MSELoss")
    parser.add_argument("-b", "--batch_size", default=16)
    parser.add_argument("-n", "--n_epochs", default=60)
    parser.add_argument("-l", "--learning_rate", default=1e-4)
    parser.add_argument("-w", "--weight_decay", default=0.05)
    parser.add_argument("--patience", default=40)
    args = parser.parse_args()
    return args


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

        instance = create_data_instance(graph, y_bus, y_gen, y_line)
        data.append(instance)

    return data

def load_data(train_dir, val_dir, test_dir):
    try:
        train = read_from_pkl("./data_generation/loaded_data/train.pkl")
        val = read_from_pkl("./data_generation/loaded_data/val.pkl")
        test = read_from_pkl("./data_generation/loaded_data/test.pkl")
        print("Data Loaded from pkl files")
    except:
        print("Data not found, loading from json files...")
        print("Training Data...")
        train = load_data_helper(train_dir)
        print("Validation Data...")
        val = load_data_helper(val_dir)
        print("Testing Data...")
        test = load_data_helper(test_dir)

        # create folder if it doesn't exist
        if not os.path.exists("./data_generation/loaded_data"):
            os.makedirs("./data_generation/loaded_data")

        # save data to pkl
        write_to_pkl(train, "./data_generation/loaded_data/train.pkl")
        write_to_pkl(val, "./data_generation/loaded_data/val.pkl")
        write_to_pkl(test, "./data_generation/loaded_data/test.pkl")

        print("Data Loaded and saved to pkl files")

    return train, val, test


def normalize_data(train, val, test):
    # train, val and test are lists of torch_geometric.data.Data objects
    # create a tensor for x, y and edge_attr for all data (train, val, test)
    combined_x = th.cat([data.x for data in train + val + test], dim=0)
    combined_y = th.cat([data.y for data in train + val + test], dim=0)
    combined_edge_attr = th.cat([data.edge_attr for data in train + val + test], dim=0)

    # Use min max normalization to normalize data between 0 and 1 
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

    epsilon = 1e-7  # to avoid division by zero
    # normalize data
    for data in train + val + test:
        data.x = (data.x - min_x) / (max_x - min_x + epsilon)
        data.y = (data.y - min_y) / (max_y - min_y + epsilon)
        data.edge_attr = (data.edge_attr - min_edge_attr) / (max_edge_attr - min_edge_attr + epsilon)

    return train, val, test




# return a torch_geometric.data.Data object for each instance
def create_data_instance(graph, y_bus, y_gen, y_line):
    g = ppl.create_nxgraph(graph, include_trafos=False)

    # https://pandapower.readthedocs.io/en/latest/elements/gen.html
    gen = graph.gen[['bus', 'p_mw', 'vm_pu']]
    gen.rename(columns={'p_mw': 'p_mw_gen'}, inplace=True)
    gen.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/load.html
    load = graph.load[['bus', 'p_mw', 'q_mvar']]
    load.rename(columns={'p_mw': 'p_mw_load'}, inplace=True)
    load.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/bus.html
    node_feat = graph.bus[['vn_kv', 'max_vm_pu', 'min_vm_pu']]

    # make sure all nodes (bus, gen, load) have the same number of features (namely the union of all features)
    node_feat = node_feat.merge(gen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(load, left_index=True, right_index=True, how='outer')
    # fill missing feature values with 0
    node_feat.fillna(0.0, inplace=True)
    # remove duplicate columns/indices
    node_feat = node_feat[~node_feat.index.duplicated(keep='first')]

    for node in node_feat.itertuples():
        # set each node features
        g.nodes[node.Index]['x'] = [float(node.vn_kv), #bus, the grid voltage level.
                                    float(node.p_mw_gen), #gen, the active power of the generator
                                    float(node.vm_pu), #gen, the voltage magnitude of the generator.
                                    float(node.p_mw_load), #load, the active power of the load
                                    float(node.q_mvar)] #load, the reactive power of the load
        
        # set each node label
        g.nodes[node.Index]['y'] = [float(y_bus['p_mw'][node.Index]), # the active power at the node
                                    float(y_bus['q_mvar'][node.Index]), # the reactive power at the node
                                    float(y_bus['va_degree'][node.Index]), # the voltage angle at the node
                                    float(y_bus['vm_pu'][node.Index])] # the voltage magnitude at the node
        
    # quit()
    # https://pandapower.readthedocs.io/en/latest/elements/line.html
    for edges in graph.line.itertuples():
        g.edges[edges.from_bus, edges.to_bus, ('line', edges.Index)]['edge_attr'] = [float(edges.r_ohm_per_km), #  line resistance in ohm per km
                                                                                     float(edges.x_ohm_per_km), #  line reactance in ohm per km
                                                                                     float(edges.c_nf_per_km), # line capacitance (line-to-earth) in nano Farad per km
                                                                                     float(edges.g_us_per_km), # dielectric conductance in micro Siemens per km
                                                                                     float(edges.max_i_ka), # maximum thermal current in kilo Ampere
                                                                                     float(edges.parallel), #  number of parallel line systems
                                                                                     float(edges.df), # derating factor: maximal current of line in relation to nominal current of line (from 0 to 1)
                                                                                     float(edges.length_km),] # The line length in km

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

    

def get_optim(optim_name):
    if optim_name == "Adam":
        return th.optim.Adam
    

def get_criterion(criterion_name):
    if criterion_name == "MSELoss":
        return nn.MSELoss()
    if criterion_name == "L1Loss":
        return nn.L1Loss()
    

def save_model(model, model_name, model_class_name):
    state = {
        'model': model, # save the model object with some of its parameters
        'state_dict': model.state_dict(),
    }

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    model_name = model_name + "_" + model_class_name + "_" + str(timestamp)

    th.save(state, f"./trained_models/{model_name}")


def plot_losses(losses, val_losses, model_name):
    epochs = np.arange(len(losses))

    plt.subplot(1, 2, 1)
    plt.title(f"{model_name} - Power Flow Training Learning Curve")
    plt.plot(epochs, losses, label="Training Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.subplot(1, 2, 2)
    plt.title(f"{model_name} - Power Flow Validation Learning Curve")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.tight_layout()
    plt.show()

def write_to_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def read_from_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data