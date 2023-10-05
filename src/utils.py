import argparse
import matplotlib.pyplot as plt
from models.GAT import GAT
from models.MessagePassing import MessagePassingGNN
from models.GraphSAGE import GraphSAGE
import os
import pandapower.plotting as ppl
import pandas as pd
import pandapower as pp
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
    
    parser.add_argument("gnn", choices=["GAT", "MessagePassing", "GraphSAGE"], default="GAT")
    parser.add_argument("--train", default="./Data/expand")
    parser.add_argument("--val", default="./Data/val")
    parser.add_argument("--test", default="./Data/test")
    parser.add_argument("-s", "--save_model", action="store_true", default=True)
    parser.add_argument("-m", "--model_name", default=''.join([random.choice(string.ascii_letters + string.digits) for _ in range(8)]))
    parser.add_argument("-p", "--plot", action="store_true", default=True)
    parser.add_argument("-o", "--optimizer", default="Adam")
    parser.add_argument("-c", "--criterion", default="MSELoss")
    parser.add_argument("-b", "--batch_size", default=16)
    parser.add_argument("-n", "--n_epochs", default=250)
    parser.add_argument("-l", "--learning_rate", default=1e-4)
    parser.add_argument("-w", "--weight_decay", default=0.05)
    args = parser.parse_args()
    return args


def load_data(dir):
    graph_path = f"{dir}/x"
    sol_path = f"{dir}/y"
    graph_paths = sorted(os.listdir(graph_path))#[:10]
    sol_paths = sorted(os.listdir(sol_path))#[:40]
    data = []

    for i, g in tqdm.tqdm(enumerate(graph_paths)):
        graph = pp.from_json(f"{graph_path}/{g}")
        y_bus = pd.read_csv(f"{sol_path}/{sol_paths[i * 3]}", index_col=0)
        y_gen = pd.read_csv(f"{sol_path}/{sol_paths[i * 3 + 1]}", index_col=0)
        y_line = pd.read_csv(f"{sol_path}/{sol_paths[i * 3 + 2]}", index_col=0)

        instance = create_data_instance(graph, y_bus, y_gen, y_line)
        data.append(instance)

    return data


# return a torch_geometric.data.Data object for each instance
def create_data_instance(graph, y_bus, y_gen, y_line):
    g = ppl.create_nxgraph(graph, include_trafos=False)

    gen = graph.gen[['bus', 'p_mw', 'vm_pu']]
    gen.rename(columns={'p_mw': 'p_mw_gen'}, inplace=True)
    gen.set_index('bus', inplace=True)

    load = graph.load[['bus', 'p_mw', 'q_mvar']]
    load.rename(columns={'p_mw': 'p_mw_load'}, inplace=True)
    load.set_index('bus', inplace=True)

    node_feat = graph.bus[['vn_kv', 'max_vm_pu', 'min_vm_pu']]
    # make sure all nodes have the same number of features
    node_feat = node_feat.merge(gen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(load, left_index=True, right_index=True, how='outer')
    # fill missing feature values with 0
    node_feat.fillna(0.0, inplace=True)
    # remove duplicate columns/indices
    node_feat = node_feat[~node_feat.index.duplicated(keep='first')]

    for node in node_feat.itertuples():
        # set each node features
        g.nodes[node.Index]['x'] = [float(node.vn_kv), #bus
                                    float(node.p_mw_gen), #gen
                                    float(node.vm_pu), #gen
                                    float(node.p_mw_load), #load
                                    float(node.q_mvar)] #load
        
        # set each node label
        g.nodes[node.Index]['y'] = [float(y_bus['p_mw'][node.Index]),
                                    float(y_bus['q_mvar'][node.Index]),
                                    float(y_bus['va_degree'][node.Index]),
                                    float(y_bus['vm_pu'][node.Index])]
        
    # quit()

    for edges in graph.line.itertuples():
        g.edges[edges.from_bus, edges.to_bus, ('line', edges.Index)]['edge_attr'] = [float(edges.r_ohm_per_km),
                                                                                     float(edges.x_ohm_per_km),
                                                                                     float(edges.c_nf_per_km),
                                                                                     float(edges.g_us_per_km),
                                                                                     float(edges.max_i_ka),
                                                                                     float(edges.parallel),
                                                                                     float(edges.df),
                                                                                     float(edges.length_km),]

    return from_networkx(g)


def get_gnn(gnn_name):
    if gnn_name == "GAT":
        return GAT
    
    if gnn_name == "MessagePassing":
        return MessagePassingGNN
    
    if gnn_name == "GraphSAGE":
        return GraphSAGE
    

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
    # timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    model_name = model_name + "_" + model_class_name # + "_" + str(timestamp)
    th.save(model.state_dict(), f"./trained_models/{model_name}.pt")

def load_model(gnn_type, path, data):
    input_dim = data[0].x.shape[1]
    edge_attr_dim = data[0].edge_attr.shape[1] 
    output_dim = data[0].y.shape[1]
    print(input_dim, edge_attr_dim, output_dim)
    gnn_class = get_gnn(gnn_type)
    model = gnn_class(input_dim, output_dim, edge_attr_dim)
    model.load_state_dict(th.load(path))
    return model


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