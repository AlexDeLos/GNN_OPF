import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import pandapower as pp
import torch as th
from torch_geometric.loader import DataLoader
import tqdm
import os
import sys
# local imports
# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_plot import distance_plot
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
        _, distance_losses = test_hetero(model, data, calc_power_vals=args.model_load_data_path is not None, save=args.no_save, path=args.save_path, name=args.save_name, plot_node_error=args.plot_node_error)
        if args.plot_node_error:
            plot_distance_losses(distance_losses, model=model)
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
    parser.add_argument("--calc_power", action="store_true", default=False)
    parser.add_argument("--no_save", action="store_false", default=True)
    parser.add_argument("--save_path", default='./Data/results')
    parser.add_argument("--save_name", default="results")
    parser.add_argument("--plot_node_error", action="store_true", default=False)
    args = parser.parse_args()
    return args


def test(model, data, calc_power_vals=False):
    """
    Test the given model on the provided data.

    Args:
        model (torch.nn.Module): The model to test.
        data (torch_geometric.data.Data): The data to use for testing.
        calc_power_vals (bool, optional): Whether to calculate power values from the output. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays: the errors and the percentage errors.
    """
    # disable scientifc notation
    np.set_printoptions(suppress=True)
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

  
def test_hetero(model, data, calc_power_vals, save, path, name, plot_node_error=False):
    """
    Test the given model on the provided data and calculate the error between the predicted and actual values.
    
    Args:
    - model: The model to be tested.
    - data: The data to be used for testing.
    - calc_power_vals: A boolean flag indicating whether to calculate power values from fixed and predicted voltages.
    - save: A boolean flag indicating whether to save the results to a CSV file.
    - path: The path where the CSV file should be saved.
    - name: The name of the CSV file.
    
    Returns:
    - error_dict: A dictionary containing the error for each node type.
    """
    np.set_printoptions(suppress=True)
    loader = DataLoader(data)
    error_dict = {
        'load': [],
        'gen': [],
        'load_gen': [],
        'ext': []
    }
    dims_dict = None
    count = 0
    max_distance = 0
    min_distance = 100
    distance_losses = []
    for g in loader:
        # Get output dims of test set
        if dims_dict is None:
            dims_dict = {node_type: g.y_dict[node_type].shape[1] for node_type in g.y_dict.keys()}

        # Outputs only voltage mag/degree predictions (when required depending o node type)
        out = model(g.x_dict, g.edge_index_dict, g.edge_attr_dict)


        if plot_node_error and count < 10:
            count += 1  
            distance_loss,l = distance_plot(model, g, hetero=True)
            if max_distance < l:
                max_distance = l
            if min_distance > l:
                min_distance = l
            # print(l)
            distance_losses.append(distance_loss)
        


        # Calculate power values from fixed and predicted voltages. Dict of tensors([p_mw, q_mvar]) per node type.
        if calc_power_vals:
            # If using physics model, then need to remove power predictions when passing to power_from_voltages_hetero
            if out['gen'].shape[1] > 1:
                out['gen'] = out['gen'][:, 1].reshape((-1, 1))
                out['load_gen'] = out['load_gen'][:, 1].reshape((-1, 1))

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

    #make all arrays the same shape
    for dis in distance_losses:
        while len(dis) > min_distance:
            dis.pop()

    distance_losses = np.mean(distance_losses,0)

    df_data = {}
    va_arr = []
    q_mvar_arr = []
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
        
        if save:
            length = len(v)
            within = np.array([np.sum(abs(v) < i, axis=0) / length for i in range(1, 101)])
            if k == 'load':
                vm_pu_within = within[:, 1]
                va_degree_within = within[:, 0]
                df_data[f'{k}_vm_pu'] = vm_pu_within
                df_data[f'{k}_va_deg'] = va_degree_within
                va_arr.append(v[:, 0])

            elif k == 'gen' or k == 'load_gen':
                va_degree_within = within[:, 1]
                df_data[f'{k}_va_deg'] = va_degree_within
                va_arr.append(v[:, 1])
                q_mvar_arr.append(v[:, 0])
    if save:
        va = np.concatenate(va_arr)
        q_mvar = np.concatenate(q_mvar_arr)
        length = len(va)
        va_within = np.array([np.sum(abs(va) < i, axis=0) / length for i in range(1, 101)])
        q_mvar_within = np.array([np.sum(abs(q_mvar) < i, axis=0) / length for i in range(1, 101)])
        df_data['va_degree'] = va_within
        df_data['q_mvar'] = q_mvar_within

        

    if save:
        df = pd.DataFrame(data=df_data)
        Path(f"{path}/").mkdir(parents=True, exist_ok=True)
        df.to_csv(f'{path}/{name}.csv')

    return error_dict, distance_losses

def plot_distance_losses(distance_losses, model):
        plt.plot(list(range(0,len(distance_losses))), distance_losses, color ='maroon')
        plt.title("Error with distance from the Slack line")
        plt.ylabel("Error")
        plt.xticks(range(0,len(distance_losses)))
        plt.xlabel("Nodes away from the Slack line the node was located")
        
        # if file is moved in another directory level relative to the root (currently in root/utils/src), this needs to be changed
        root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        plot_directory = root_directory + "/plots"
        if not os.path.exists(plot_directory):
            os.mkdir(plot_directory)

        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

        model_name = "distance_plot" + "_" + model.class_name + "_" + str(timestamp)
        plt.savefig(f"{plot_directory}/{model_name}.png", format="png")
        plt.show()

def plot_within(data, title):
    plt.plot(range(1, 101), data, color='red')
    plt.title(title)
    plt.xlabel("Error Threshold in %")
    plt.ylabel("Percent within error threshold")
    plt.show()

def normalize_test():
    graph_path = f"Data/bfs_gen/large/x"
    graph_paths = sorted(os.listdir(graph_path))

    for g in tqdm.tqdm(graph_paths):
        graph = pp.from_json(f"{graph_path}/{g}")
        print(graph.line['r_ohm_per_km'].min())
        print(graph.line['r_ohm_per_km'].max())


if __name__ == '__main__':
    main()