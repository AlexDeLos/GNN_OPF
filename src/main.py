import argparse
import os
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import random
import string

import networkx as nx
import pandapower as pp
import pandapower.plotting as ppl
from architectures.GAT import GATNodeRegression

from models.GATConv import GATConvolution
from models.MessagePassingConv import MessagePassingGNN

import torch as th
import torch.nn as nn
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader as pyg_DataLoader

import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    print("Parsing Arguments")
    arguments = get_arguments()
    print(f"Parsed arguments: {arguments}")

    print("Loading Training Data")
    train = load_data(arguments.train)
    print("Loading Validation Data")
    val = load_data(arguments.val)
    print("Loading Testing Data")
    test = load_data(arguments.test)

    print(f"Data Loaded \n",
          f"Number of training samples = {len(train)}\n",
          f"Number of validation samples = {len(val)}\n",
          f"Number of testing samples = {len(test)}\n",)

    print("Training Model")
    model, losses, val_losses = train_model(arguments, train, val, test)
    if arguments.save_model:
        print("Saving Model")
        save_model(model, arguments.model_name)
    if arguments.plot:
        plot_losses(losses, val_losses)
    

def get_arguments():
    parser = argparse.ArgumentParser(prog="GNN script",
                                     description="Run a GNN to solve an inductive power system problem (power flow only for now)")
    parser.add_argument("gnn", choices=["GATConv", "GAT", "MessagePassing"])
    parser.add_argument("--train", default="./Data/train")
    parser.add_argument("--val", default="./Data/val")
    parser.add_argument("--test", default="./Data/test")
    parser.add_argument("-s", "--save_model", action="store_true", default=True)
    parser.add_argument("-m", "--model_name", default=''.join([random.choice(string.ascii_letters + string.digits) for _ in range(8)]))
    parser.add_argument("-p", "--plot", action="store_true", default=True)
    parser.add_argument("-o", "--optimizer", default="Adam")
    parser.add_argument("-c", "--criterion", default="MSELoss")
    parser.add_argument("-b", "--batch_size", default=16)
    parser.add_argument("-n", "--n_epochs", default=150)
    parser.add_argument("-l", "--learning_rate", default=1e-5)
    parser.add_argument("-w", "--weight_decay", default=0.05)
    args = parser.parse_args()
    return args


def load_data(dir):
    graph_path = f"{dir}/x"
    sol_path = f"{dir}/y"
    graph_paths = sorted(os.listdir(graph_path))
    sol_paths = sorted(os.listdir(sol_path))
    data = []
    for i, g in enumerate(graph_paths):
        #print(g)
        #print(sol_paths[3 * i])
        graph = pp.from_json(f"{graph_path}/{g}")
        y_bus = pd.read_csv(f"{sol_path}/{sol_paths[i * 3]}")
        y_gen = pd.read_csv(f"{sol_path}/{sol_paths[i * 3 + 1]}")
        y_line = pd.read_csv(f"{sol_path}/{sol_paths[i * 3 + 2]}")

        instance = create_data_instance(graph, y_bus, y_gen, y_line, g == 'case118_12_eXGwM8p1.json')
        data.append(instance)

    return data

# return a torch_geometric.data.Data object for each instance
def create_data_instance(graph, y_bus, y_gen, y_line, p):
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

    for i, node in enumerate(node_feat.itertuples()):
        # print("Indices")
        # print(i, node.Index)
        # set each node features
        g.nodes[node.Index]['x'] = [float(node.vn_kv), 
                                    float(node.max_vm_pu), 
                                    float(node.min_vm_pu),
                                    float(node.p_mw_gen),
                                    float(node.vm_pu),
                                    float(node.p_mw_load),
                                    float(node.q_mvar)]
        # set the nodes labels
        g.nodes[node.Index]['y'] = [float(y_bus['vm_pu'][i])]
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
    if gnn_name == "GATConv":
        return GATConvolution
    
    if gnn_name == "MessagePassing":
        return MessagePassingGNN
    
def get_optim(optim_name):
    if optim_name == "Adam":
        return th.optim.Adam
    
def get_criterion(criterion_name):
    if criterion_name == "MSELoss":
        return nn.MSELoss()
    
def train_model(arguments, train, val, test):
    input_dim = train[0].x.shape[1]
    edge_attr_dim = train[0].edge_attr.shape # why not [1]
    output_dim = train[0].y.shape[1]

    print(f"Input shape: {input_dim}\nOutput shape: {output_dim}")

    batch_size = arguments.batch_size
    train_dataloader = pyg_DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = pyg_DataLoader(val, batch_size=batch_size, shuffle=True)
    gnn_class = get_gnn(arguments.gnn)
    gnn = gnn_class(input_dim, output_dim, edge_attr_dim)
    print(f"GNN: \n{gnn}")

    optimizer_class = get_optim(arguments.optimizer)
    optimizer = optimizer_class(gnn.parameters(), lr=arguments.learning_rate)
    criterion = get_criterion(arguments.criterion)

    losses = []
    val_losses = []

    for epoch in tqdm.tqdm(range(arguments.n_epochs)): #args epochs
        epoch_loss = 0.0
        epoch_val_loss = 0.0
        gnn.train()
        for batch in train_dataloader:
            epoch_loss += train_batch(data=batch, model=gnn, optimizer=optimizer, criterion=criterion)
        gnn.eval()
        for batch in val_dataloader:
            epoch_val_loss += evaluate_batch(data=batch, model=gnn, criterion=criterion)

        avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
        avg_epoch_val_loss = epoch_val_loss.item() / len(val_dataloader)

        losses.append(avg_epoch_loss)
        val_losses.append(avg_epoch_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.3f}, val_loss: {avg_epoch_val_loss:.3f}')
    
    return gnn, losses, val_losses

def train_batch(data, model, optimizer, criterion, device='cpu'):
    model.to(device)
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss

def evaluate_batch(data, model, criterion, device='cpu'):
    model.to(device)
    out = model(data)
    loss = criterion(out, data.y)
    return loss

def save_model(model, model_name):
    th.save(model.state_dict(), f"./trained_models/{model_name}")

def plot_losses(losses, val_losses):
    epochs = np.arange(len(losses))

    plt.subplot(1, 2, 1)
    plt.title("GNN Power Flow Training Learning Curve")
    plt.plot(epochs, losses, label="Training Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.subplot(1, 2, 2)
    plt.title("GNN Power Flow Validation Learning Curve")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()