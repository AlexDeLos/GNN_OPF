import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
import torch as th
from torch_geometric.loader import DataLoader
import tqdm
import os
import sys
# local imports
# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_model, load_model_hetero, read_from_pkl
from utils.utils_homo import normalize_data
from utils.utils_hetero import normalize_data_hetero


def main():
    args = parse_args()
    data = read_from_pkl(f"{args.data_path}/pickled.pkl")
    if args.normalize:
        print("Normalizing Data")
        if args.gnn_type[:6] != "Hetero":
            data, _, _ = normalize_data(data, data, data)
        else: 
            data, _, _ = normalize_data_hetero(data, data, data)
    if "HeteroGNN" in args.model_path:
        model = load_model_hetero(args.gnn_type, args.model_path, data)
    else:
        model = load_model(args.gnn_type, args.model_path, data)
    model.eval()
    if "HeteroGNN" in args.model_path:
        test_hetero(model, data)
    else:
        test(model, data)

def parse_args():
    parser = argparse.ArgumentParser("Testing powerfloww GNN models")
    parser.add_argument("-g", "--gnn_type", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-d", "--data_path", required=True)
    parser.add_argument("--n_hidden_gnn", default=1, type=int)
    parser.add_argument("--gnn_hidden_dim", default=16, type=int)
    parser.add_argument("--n_hidden_lin", default=0, type=int)
    parser.add_argument("--lin_hidden_dim", default=32, type=int)
    parser.add_argument("--normalize", action='store_true', default=False)
    parser.add_argument("--no_linear", action="store_true", default=False)
    args = parser.parse_args()
    return args


def test(model, data):
    print("testing")
    loader = DataLoader(data)
    errors = []
    p_errors = []
    first = True
    for g in loader:
        out = model(g)
        error = th.abs(th.sub(g.y, out))
        p_error = th.div(error, g.y) * 100
        errors.append(error.detach().numpy())
        p_errors.append(p_error.detach().numpy())
    errors = np.concatenate(errors)
    errors = errors.reshape((-1, 2))
    print(errors.shape, np.shape(errors), "shape of errors")

    p_errors = np.concatenate(p_errors)
    errors = errors.reshape((-1, 2))
    print(errors.shape, np.shape(errors), "shape of errors")

    mask = np.isinf(p_errors)
    p_errors[mask] = 0

    print("within 0.1%", np.sum(abs(p_errors) < 0.1, axis=0) / len(p_errors))
    print("within 0.5%", np.sum(abs(p_errors) < 0.5, axis=0) / len(p_errors))
    print("within 1%", np.sum(abs(p_errors) < 1, axis=0) / len(p_errors))
    print("within 2%", np.sum(abs(p_errors) < 2, axis=0) / len(p_errors))
    print("within 5%", np.sum(abs(p_errors) < 5, axis=0) / len(p_errors))
    print("within 10%", np.sum(abs(p_errors) < 10, axis=0) / len(p_errors))
    print("within 15%", np.sum(abs(p_errors) < 15, axis=0) / len(p_errors))
    print("within 25%", np.sum(abs(p_errors) < 25, axis=0) / len(p_errors))
    print("within 50%", np.sum(abs(p_errors) < 50, axis=0) / len(p_errors))

    return errors, p_errors

def test_hetero(model, data):
    loader = DataLoader(data)
    error_dict = {
        'load': [],
        'gen': [],
        'load_gen': [],
        'ext': []
    }
    for g in loader:
        out = model(g.x_dict, g.edge_index_dict, g.edge_attr_dict)
        for node_type, y in g.y_dict.items():
            error = th.abs(th.sub(out[node_type], y))
            p_error = th.div(error, y) * 100
            error_dict[node_type].append(p_error.detach().numpy())

    error_dict['load'] = np.concatenate(error_dict['load']).reshape((-1, 2))
    error_dict['gen'] = np.concatenate(error_dict['gen']).reshape((-1, 1))
    error_dict['load_gen'] = np.concatenate(error_dict['load_gen']).reshape((-1, 1))

    for k, v in error_dict.items():
        if k == 'ext':
            continue
        print(k, len(v))
        print("within 0.1%", np.sum(abs(v) < 0.1, axis=0) / len(v))
        print("within 0.5%", np.sum(abs(v) < 0.5, axis=0) / len(v))
        print("within 1%", np.sum(abs(v) < 1, axis=0) / len(v))
        print("within 2%", np.sum(abs(v) < 2, axis=0) / len(v))
        print("within 5%", np.sum(abs(v) < 5, axis=0) / len(v))
        print("within 10%", np.sum(abs(v) < 10, axis=0) / len(v))
        print("within 15%", np.sum(abs(v) < 15, axis=0) / len(v))
        print("within 25%", np.sum(abs(v) < 25, axis=0) / len(v))
        print("within 50%", np.sum(abs(v) < 50, axis=0) / len(v))

    return error_dict

def normalize_test():
    graph_path = f"Data/bfs_gen/large/x"
    graph_paths = sorted(os.listdir(graph_path))

    for g in tqdm.tqdm(graph_paths):
        graph = pp.from_json(f"{graph_path}/{g}")
        print(graph.line['r_ohm_per_km'].min())
        print(graph.line['r_ohm_per_km'].max())


if __name__ == '__main__':
    main()