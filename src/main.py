import argparse
import os
import pandas as pd
import tqdm

import pandapower as pp
import pandapower.plotting as ppl

from models.GCN import GCN

import torch as th
import torch.nn as nn
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader as pyg_DataLoader

def main():
    print("Loading Data")
    data = load_data()
    print(f"Data Loaded \nNumber of samples = {len(data)}")
    print("Parsing Arguments")
    arguments = get_arguments()
    print(f"Parsed arguments: {arguments}")
    train_model(arguments, data)

def get_arguments():
    parser = argparse.ArgumentParser(prog="GNN script",
                                     description="Run a GNN to solve an inductive power system problem (power flow only for now)")
    parser.add_argument("gnn", choices=["GCN"])
    parser.add_argument("-o", "--optimizer", default="Adam")
    parser.add_argument("-c", "--criterion", default="MSELoss")
    parser.add_argument("-b", "--batch_size", default=16)
    parser.add_argument("-n", "--n_epochs", default=11)
    parser.add_argument("-l", "--learning_rate", default=1e-3)
    parser.add_argument("-w", "--weight_decay", default=0)
    args = parser.parse_args()
    return args


def load_data():
    data_dir = './Data/data'
    graph_path = f"{data_dir}/x"
    sol_path = f"{data_dir}/y"
    graph_paths = sorted(os.listdir(graph_path))
    sol_paths = sorted(os.listdir(sol_path))
    data = []
    for i, g in enumerate(graph_paths):
        graph = pp.from_json(f"{graph_path}/{g}")
        y_bus = pd.read_csv(f"{sol_path}/{sol_paths[i * 3]}")
        y_gen = pd.read_csv(f"{sol_path}/{sol_paths[i * 3 + 1]}")
        y_line = pd.read_csv(f"{sol_path}/{sol_paths[i * 3 + 2]}")

        data.append(create_data_instance(graph, y_bus, y_gen, y_line))
    return data

def create_data_instance(graph, y_bus, y_gen, y_line):
    g = ppl.create_nxgraph(graph)
    for i, node in enumerate(graph.bus.itertuples()):
        g.nodes[node.Index]['x'] = [node.vn_kv,node.max_vm_pu,node.min_vm_pu]
        g.nodes[node.Index]['y'] = [float(y_bus['p_mw'][i]), float(y_bus['va_degree'][i])]
    
    for edges in graph.line.itertuples():
        g.edges[edges.from_bus, edges.to_bus, ('line', edges.Index)]['edge_attr'] = [edges.r_ohm_per_km, edges.length_km]

    return from_networkx(g)

def get_gnn(gnn_name):
    if gnn_name == "GCN":
        return GCN
    
def get_optim(optim_name):
    if optim_name == "Adam":
        return th.optim.Adam
    
def get_criterion(criterion_name):
    if criterion_name == "MSELoss":
        return nn.MSELoss()
    
def train_model(arguments, data):
    input_dim = data[0].x.shape[1]
    edge_attr_shape = data[0].edge_attr.shape
    output_dim = data[0].y.shape[1]

    batch_size = arguments.batch_size
    dataloader = pyg_DataLoader(data, batch_size=batch_size, shuffle=True) # args batch size
    gnn_class = get_gnn(arguments.gnn)
    gnn = gnn_class(input_dim, output_dim)
    print(f"GNN: \n{gnn}")

    optimizer_class = get_optim(arguments.optimizer)
    optimizer = optimizer_class(gnn.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay) # args optim, lr, weight_decay
    criterion = get_criterion(arguments.criterion)

    losses = []
    val_losses = []

    for epoch in tqdm.tqdm(range(arguments.n_epochs)): #args epochs
        epoch_loss = 0.0
        epoch_val_loss = 0.0

        for batch in dataloader:
            epoch_loss += train_batch(data=batch, model=gnn, optimizer=optimizer, criterion=criterion)
            epoch_val_loss += evaluate_batch(data=batch, model=gnn, criterion=criterion)

        losses.append(epoch_loss / batch_size)
        val_losses.append(epoch_val_loss / batch_size)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, trn_Loss: {epoch_loss:.3f}, val_loss: {epoch_val_loss:.3f}')

def train_batch(data, model, optimizer, criterion, device='cpu'):
    model.train()
    model.to(device)
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss

def evaluate_batch(data, model, criterion, device='cpu'):
    model.eval()
    model.to(device)
    out = model(data)
    loss = criterion(out, data.y)
    return loss


if __name__ == "__main__":
    main()