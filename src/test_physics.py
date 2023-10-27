import os
import argparse
import numpy as np
import torch as th
from torch_geometric.loader import DataLoader
from utils.utils import load_model, load_model_hetero, read_from_pkl
from utils.utils_hetero import physics_loss_hetero, power_from_voltages_hetero


def main():
    args = parse_args()

    root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_folder = root_directory + f"/GNN_OPF/{args.model_path}"

    data_folder = root_directory + f"/GNN_OPF/{args.data_path}"
    data = read_from_pkl(data_folder + "/pickled.pkl")
    if "HeteroGNN" in args.gnn_type:
        model = load_model_hetero(args.gnn_type, model_folder, data)
    else:
        model = load_model(args.gnn_type, model_folder, data)
    model.eval()
    if "HeteroGNN" in args.gnn_type:
        test_hetero_physics(model, data, calc_power_vals=args.power_from_volt)
    else:
        print("not available yet")
        # test_physics(model, data)

def parse_args():
    parser = argparse.ArgumentParser("Testing powerflow GNN models")
    parser.add_argument("-g", "--gnn_type", required=True)  # HeteroGNN
    parser.add_argument("-m", "--model_path", required=True)  # trained_models/HeteroGINE_physics_rnd_wlk_smalldim_moreconv/HeteroGINE_physics_rnd_wlk-HeteroGNN.pt
    parser.add_argument("-d", "--data_path", required=True)  # Data_phys_rnd_walk/test
    parser.add_argument("--power_from_volt", action="store_true", default=False)
    args = parser.parse_args()
    return args


def test_hetero_physics(model, data, calc_power_vals=False):
    loader = DataLoader(data)
    imbalances_dict = {
        'load': [],
        'gen': [],
        'load_gen': [],
        'ext': []
    }
    error_dict = {
        'load': [],
        'gen': [],
        'load_gen': [],
        'ext': []
    }
    node_count_dict = None

    i = 0
    for g in loader:
        i += 1
        if i % 50 == 0:
            print(f"Graph: {i}")

        if node_count_dict is None:
            node_count_dict = {node_type: g.x_dict[node_type].shape[0] for node_type in g.x_dict.keys()}

        # Outputs all missing values per node type
        out = model(g.x_dict, g.edge_index_dict, g.edge_attr_dict)

        # Throw away the (bad) power predictions, and calculate from the better voltage predictions
        if calc_power_vals:
            # Keep only predicted vm_pu, remove p_mw
            out['gen'] = out['gen'][:, 1].reshape((-1, 1))
            out['load_gen'] = out['load_gen'][:, 1].reshape((-1, 1))

            power_values = power_from_voltages_hetero(g, out)
            for node_type in g.y_dict.keys():
                if node_type == 'gen' or node_type == 'load_gen':
                    out[node_type] = th.cat((power_values[node_type][:, 1].reshape(-1, 1), out[node_type].reshape((-1, 1))), 1)
                elif node_type == 'ext':
                    out[node_type] = power_values[node_type]

        # Calculate total loss per node type
        loss = physics_loss_hetero(g, out, log_loss=False, per_node_type=True)

        # Add total active+reactive power imbalance per node type and node counts to dicts (all tested on same graph; counts don't change)
        for node_type, y in g.y_dict.items():
            # Add graph imbalance
            imbalances_dict[node_type].append(loss[node_type].detach().numpy())

            # Add per node errors
            error = th.abs(th.sub(out[node_type], y))
            p_error = th.div(error, y) * 100
            error_dict[node_type].append(p_error.detach().numpy())

    for node_type, imbalance in imbalances_dict.items():
        node_count = node_count_dict[node_type]
        avg_imbalances_per_graph = np.stack(imbalances_dict[node_type]).reshape((-1, 1)) / node_count
        worst_imbalance = np.max(avg_imbalances_per_graph)

        print(f"\n##### {node_type} #####")
        print(f"Worst avg. imbalance per node for a graph: {worst_imbalance}")
        print(f"Overall avg. imbalance: {np.sum(avg_imbalances_per_graph) / len(loader)}\n")

    for k in error_dict.keys():
        v = np.concatenate(error_dict[k]).reshape((-1, 2))

        print(f"\n{k}, {len(v)}")
        print("within 0.1%", np.sum(abs(v) < 0.1, axis=0) / len(v))
        print("within 0.5%", np.sum(abs(v) < 0.5, axis=0) / len(v))
        print("within 1%  ", np.sum(abs(v) < 1, axis=0) / len(v))
        print("within 2%  ", np.sum(abs(v) < 2, axis=0) / len(v))
        print("within 5%  ", np.sum(abs(v) < 5, axis=0) / len(v))
        print("within 10% ", np.sum(abs(v) < 10, axis=0) / len(v))
        print("within 15% ", np.sum(abs(v) < 15, axis=0) / len(v))
        print("within 25% ", np.sum(abs(v) < 25, axis=0) / len(v))
        print("within 50% ", np.sum(abs(v) < 50, axis=0) / len(v))

    return

if __name__ == '__main__':
    main()
