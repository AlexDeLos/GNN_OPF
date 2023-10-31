import numpy as np
import os
import tqdm
import pandapower as pp

from utils.utils import load_model_hetero, read_from_pkl
from torch_geometric.loader import DataLoader
from copy import deepcopy

# load graph json
# make copy
# load graph torch geo
# make prediction torch geo
# run powerflow on graph 1
# run powerflow on graph 2 with init

DATA_PATH = './Data/final/rnd_neighbor/test'
MODEL_DATA_PATH = './Data/final/rnd_neighbor/val'
MODEL_PATH = './trained_models/physics_opt-HeteroGNN/best-physics_opt-HeteroGNN.pt'
GNN_TYPE = 'HeteroGNN'

def main():
    graph_path = f"{DATA_PATH}/x"
    graph_paths = sorted(os.listdir(graph_path))
    graphs = [pp.from_json(f"{graph_path}/{g}") for g in graph_paths]
    geo_graphs = read_from_pkl(f"{DATA_PATH}/pickled.pkl")
    loader = DataLoader(geo_graphs)

    data_model_loading = read_from_pkl(f"{MODEL_DATA_PATH}/pickled.pkl")
    model = load_model_hetero(GNN_TYPE, MODEL_PATH, data_model_loading)
    blank_iterations = []
    model_iterations = []
    for i, g in enumerate(loader):
        blank_start = graphs[i]
        model_start = deepcopy(blank_start)
        out = model(g.x_dict, g.edge_index_dict, g.edge_attr_dict)

        index_map = get_indices(blank_start)

        vm_pu_guess = np.zeros(len(blank_start.bus))
        va_degree_guess = np.zeros(len(blank_start.bus))

        for node_type, y in g.y_dict.items():
            type_index = index_map[node_type]
            if node_type == 'load':
                vm_pu_guess[type_index] = out[node_type][:, 1].detach().numpy()
                va_degree_guess[type_index] = out[node_type][:, 0].detach().numpy()
            elif node_type == 'gen' or node_type == 'load_gen': 
                vm_pu_guess[type_index] = g.x_dict[node_type][:, -1]
                va_degree_guess[type_index] = out[node_type][:, -1].detach().numpy()
            else:
                vm_pu_guess[type_index] = g.x_dict[node_type][:, 0]
                va_degree_guess[type_index] = g.x_dict[node_type][:, 1]

        blank_start_iterations = pp.runpp(blank_start, numba=False, algorithm='iwamoto_nr')
        model_start_iterations = pp.runpp(model_start, numba=False, init_vm_pu=vm_pu_guess, init_va_degree=va_degree_guess, algorithm='iwamoto_nr')
        blank_iterations.append(blank_start_iterations)
        model_iterations.append(model_start_iterations)
    print(np.mean(blank_iterations))
    print(np.mean(model_iterations))


def get_indices(graph):
    ext_index = graph.ext_grid['bus'].to_numpy()
    gen_index = graph.gen['bus'].to_numpy()
    load_index = graph.load['bus'].to_numpy()
    load_gen_index = np.intersect1d(gen_index, load_index)
    gen_index = np.setdiff1d(gen_index, load_gen_index)
    load_index = np.setdiff1d(graph.bus.index.to_numpy(), np.concatenate([load_gen_index, ext_index, gen_index]))

    index_map = {
        'load': load_index,
        'gen': gen_index,
        'load_gen': load_gen_index,
        'ext': ext_index
    }
    return index_map

if __name__ == '__main__':
    main()