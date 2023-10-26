import argparse
from models.GAT import GAT
from models.MessagePassing import MessagePassingGNN
from models.GraphSAGE import GraphSAGE
from models.GINE import GINE
from models.HeterogenousGNN import HeteroGNN
import os
import pandas as pd
import pandapower as pp
import pickle
import random
import string
import torch as th
import torch.nn as nn
import tqdm
import os
import sys
# local imports
# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_hetero import create_hetero_data_instance
from utils.utils_homo import create_data_instance
from utils.utils_physics import create_physics_data_instance

def get_arguments():
    parser = argparse.ArgumentParser(prog="GNN script",
                                     description="Run a GNN to solve an inductive power system problem (power flow only for now)")
    
    # Important: prefix all heterogeneous GNNs names with "Hetero"
    parser.add_argument("gnn", choices=["GAT", "MessagePassing", "GraphSAGE", "GINE", "HeteroGNN"], default="GAT")
    # if file is moved in another directory level relative to the root (currently in root/utils/src), this needs to be changed
    root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--train", default=root_directory + "/Data/train")
    parser.add_argument("--val", default=root_directory + "/Data/val")
    parser.add_argument("--test", default=root_directory + "/Data/test")
    parser.add_argument("-s", "--save_model", action="store_true", default=True)
    parser.add_argument("-m", "--model_name", default=''.join([random.choice(string.ascii_letters + string.digits) for _ in range(8)]))
    parser.add_argument("-p", "--plot", action="store_true", default=True)
    parser.add_argument("-o", "--optimizer", default="Adam")
    parser.add_argument("-c", "--criterion", default="MSELoss")
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument("-n", "--n_epochs", default=200, type=int)
    parser.add_argument("-l", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("-w", "--weight_decay", default=0.05, type=float)
    parser.add_argument("--mixed_loss_weight", default=0.1, type=float)
    parser.add_argument("--n_hidden_gnn", default=2, type=int)
    parser.add_argument("--gnn_hidden_dim", default=32, type=int)
    parser.add_argument("--n_hidden_lin", default=2, type=int)
    parser.add_argument("--lin_hidden_dim", default=8, type=int)
    parser.add_argument("--patience", default=40, type=int)
    parser.add_argument("--plot_node_error", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--no_linear", action="store_true", default=False)
    parser.add_argument("--loss_type", choices=['standard', 'physics', 'mixed'], default='standard')
    parser.add_argument("--value_mode", choices=['all', 'missing', 'voltage'], default='all')
    parser.add_argument("--pretrain", action="store_true", default=False)

    args = parser.parse_args()
    return args


def load_data(train_dir, val_dir, test_dir, gnn_type, missing=False, volt=False, physics_data=False):
    try:
        train = read_from_pkl(f"{train_dir}/pickled.pkl")
    except:
        print("Training data not found, loading from json files...")
        train = load_data_helper(train_dir, gnn_type, missing=missing, volt=volt, physics_data=physics_data)
        # save data to pkl
        write_to_pkl(train, f"{train_dir}/pickled.pkl")
        print("Training data loaded and saved to pkl file")

    try:
        val = read_from_pkl(f"{val_dir}/pickled.pkl")
    except:
        print("Validation data not found, loading from json files...")
        val = load_data_helper(val_dir, gnn_type, missing=missing, volt=volt, physics_data=physics_data)
        write_to_pkl(val, f"{val_dir}/pickled.pkl")
        print("Validation data loaded and saved to pkl file")

    try:
        test = read_from_pkl(f"{test_dir}/pickled.pkl")
    except:
        print("Test data not found, loading from json files...")
        # Physics_data true and missing/volt to false for testing sets, because we also need the ground truth power values, not just voltages
        test = load_data_helper(test_dir, gnn_type, missing=False, volt=False, physics_data=True)
        write_to_pkl(test, f"{test_dir}/pickled.pkl")
        print("Test data loaded and saved to pkl file")
        
    return train, val, test

  
def load_data_helper(dir, gnn_type, missing=False, volt=False, physics_data=False):
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
            instance = create_hetero_data_instance(graph, y_bus, physics_data=physics_data)
        data.append(instance)

    return data


def get_gnn(gnn_name):
    if gnn_name == "GAT":
        return GAT
    if gnn_name == "MessagePassing":
        return MessagePassingGNN
    if gnn_name == "GraphSAGE":
        return GraphSAGE
    if gnn_name == "GINE":
        return GINE
    if gnn_name == "HeteroGNN":
        return HeteroGNN


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
        return th.optim.RMSprop
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


def save_model(model, model_name, epoch=None):
    model_name = model_name + "-" + model.class_name
    # if file is moved in another directory level relative to the root (currently in root/utils/src), this needs to be changed
    root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_directory = root_directory + "/trained_models/" + model_name
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    if epoch is not None:
        th.save(model.state_dict(), f"{save_directory}/{model_name}_epoch_{epoch}.pt")
    else:
        th.save(model.state_dict(), f"{save_directory}/{model_name}_final.pt")
    
    
def load_model(gnn_type, path, data):
    input_dim = data[0].x.shape[1]
    edge_attr_dim = data[0].edge_attr.shape[1]
    output_dim = data[0].y.shape[1]
    gnn_class = get_gnn(gnn_type)
    model = gnn_class(input_dim,
                      output_dim, 
                      edge_attr_dim,
                    )
    print(model)
    model.load_state_dict(th.load(path))
    return model


def load_model_hetero(gnn_type, path, data):
    output_dims = {node_type: data[0].y_dict[node_type].shape[1] for node_type in data[0].y_dict.keys()}
    gnn_class = get_gnn(gnn_type)
    gnn = gnn_class(output_dim_dict=output_dims, edge_types=data[0].edge_index_dict.keys())
    gnn.load_state_dict(th.load(path))
    return gnn


def write_to_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_from_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data