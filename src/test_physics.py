import os
import argparse
import numpy as np
from torch_geometric.loader import DataLoader
from utils.utils import load_model, load_model_hetero, read_from_pkl
from utils.utils_hetero import physics_loss_hetero


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
        test_hetero_physics(model, data)
    else:
        print("not available yet")
        # test_physics(model, data)

def parse_args():
    parser = argparse.ArgumentParser("Testing powerflow GNN models")
    parser.add_argument("-g", "--gnn_type", required=True)  # HeteroGNN
    parser.add_argument("-m", "--model_path", required=True)  # trained_models/HeteroGINE_physics_rnd_wlk_smalldim_moreconv/HeteroGINE_physics_rnd_wlk-HeteroGNN.pt
    parser.add_argument("-d", "--data_path", required=True)  # Data_phys_rnd_walk/test
    args = parser.parse_args()
    return args


def test_hetero_physics(model, data):
    loader = DataLoader(data)
    error_dict = {
        'load': [],
        'gen': [],
        'load_gen': [],
        'ext': []
    }
    node_count_dict = {
        'load': 0,
        'gen': 0,
        'load_gen': 0,
        'ext': 0
    }
    first = True
    i = 0
    for g in loader:
        i+=1
        if i % 50 == 0:
            print(f"Graph: {i}")

        # Outputs only voltage mag/degree predictions (when required depending on node type)
        out = model(g.x_dict, g.edge_index_dict, g.edge_attr_dict)

        # Calculate total loss per node type
        loss = physics_loss_hetero(g, out, log_loss=False, per_node_type=True)

        # Add total active+reactive power imbalance per node type and node counts to dicts (all tested on same graph; counts don't change)
        for node_type, _ in g.y_dict.items():
            error_dict[node_type].append(loss[node_type].detach().numpy())
            if first:
                node_count_dict[node_type] = g.x_dict[node_type].shape[0]

        first = False

    for node_type, imbalance in error_dict.items():
        node_count = node_count_dict[node_type]
        avg_imbalances_per_graph = np.stack(error_dict[node_type]).reshape((-1, 1)) / node_count
        worst_imbalance = np.max(avg_imbalances_per_graph)
        print(f"\n##### {node_type} #####")
        print(f"Worst avg. imbalance per node for a graph: {worst_imbalance}")
        print(f"Overall avg. imbalance: {np.sum(avg_imbalances_per_graph) / len(loader)}")

    return

if __name__ == '__main__':
    main()