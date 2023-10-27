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
from utils.utils_hetero import normalize_data_hetero, power_from_voltages_hetero
from utils.utils_physics import power_from_voltages


def main():
    args = parse_args()
    data = read_from_pkl(f"{args.data_path}/pickled.pkl")
    if args.model_load_data_path is not None:
        data_model_loading = read_from_pkl(f"{args.model_load_data_path}/pickled.pkl")
    else:
        data_model_loading = data
    if args.normalize:
        print("Normalizing Data")
        if args.gnn_type[:6] != "Hetero":
            data, _, _ = normalize_data(data, data, data)
        else: 
            data, _, _ = normalize_data_hetero(data, data, data)
    if "HeteroGNN" in args.gnn_type:
        model = load_model_hetero(args.gnn_type, args.model_path, data_model_loading)
    else:
        model = load_model(args.gnn_type, args.model_path, data_model_loading)
    model.eval()
    if "HeteroGNN" in args.gnn_type:
        test_hetero(model, data, calc_power_vals=args.model_load_data_path is not None)
    else:
        test(model, data, calc_power_vals=args.model_load_data_path is not None)

def parse_args():
    parser = argparse.ArgumentParser("Testing powerfloww GNN models")
    parser.add_argument("-g", "--gnn_type", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-d", "--data_path", required=True)
    # Provide train or val data path if model was trained to predict missing voltages, but testing also checks power values which have to be calculated.
    # If not provided, assumes test set has same labels as train set (power values will not be calculated).
    parser.add_argument("-l", "--model_load_data_path", required=False)
    parser.add_argument("--n_hidden_gnn", default=1, type=int)
    parser.add_argument("--gnn_hidden_dim", default=16, type=int)
    parser.add_argument("--n_hidden_lin", default=0, type=int)
    parser.add_argument("--lin_hidden_dim", default=32, type=int)
    parser.add_argument("--normalize", action='store_true', default=False)
    parser.add_argument("--no_linear", action="store_true", default=False)
    args = parser.parse_args()
    return args


def test(model, data, calc_power_vals=False):
    print("testing")
    loader = DataLoader(data)
    errors = []
    p_errors = []
    output_shape = None
    for g in loader:
        out = model(g)

        if output_shape is None:
            output_shape = g.y.shape[1]

        if calc_power_vals and out.shape[1] == 2:
            # Assumes output is [vm_pu, va_degree]
            power_values = power_from_voltages(g, out, angles_are_radians=False)
            # Out should now have [p_mw, q_mvar, vm_pu, va_degree]
            out = th.cat((power_values, out), 1)
        error = th.abs(th.sub(g.y, out))
        p_error = th.div(error, g.y) * 100
        errors.append(error.detach().numpy())
        p_errors.append(p_error.detach().numpy())

    errors = np.concatenate(errors)
    errors = errors.reshape((-1, output_shape))
    print(errors.shape, np.shape(errors), "shape of errors")

    p_errors = np.concatenate(p_errors)
    errors = errors.reshape((-1, output_shape))
    print(errors.shape, np.shape(errors), "shape of errors")

    mask = np.logical_or(np.isinf(p_errors), np.isnan(p_errors))
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

  
def test_hetero(model, data, calc_power_vals):
    loader = DataLoader(data)
    error_dict = {
        'load': [],
        'gen': [],
        'load_gen': [],
        'ext': []
    }
    dims_dict = None

    for g in loader:
        # Get output dims of test set
        if dims_dict is None:
            dims_dict = {node_type: g.y_dict[node_type].shape[1] for node_type in g.y_dict.keys()}

        # Outputs only voltage mag/degree predictions (when required depending o node type)
        out = model(g.x_dict, g.edge_index_dict, g.edge_attr_dict)

        # Calculate power values from fixed and predicted voltages. Dict of tensors([p_mw, q_mvar]) per node type.
        if calc_power_vals:
            power_values = power_from_voltages_hetero(g, out)

        # y should contain the missing 2 values per node type (which depends on node type)
        for node_type, y in g.y_dict.items():
            if calc_power_vals:
                # We only add the missing act or reactive power which should be predicted:
                #   gens miss reactive power, ext miss both
                if node_type == 'gen' or node_type == 'load_gen':
                    out[node_type] = th.cat((power_values[node_type][:, 1].reshape(-1, 1), out[node_type].reshape(-1, 1)), 1)
                elif node_type == 'ext':
                    out[node_type] = power_values[node_type]
            error = th.abs(th.sub(out[node_type], y))
            p_error = th.div(error, y) * 100
            error_dict[node_type].append(p_error.detach().numpy())

    for k in dims_dict.keys():
        v = np.concatenate(error_dict[k]).reshape((-1, dims_dict[k]))

        print(f"\n{k}, {len(v)}")
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