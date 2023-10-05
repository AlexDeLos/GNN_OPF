import torch as th
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader


from utils import get_gnn, load_data, load_model

def main():
    args = parse_args()
    data = load_data(args.data_path)
    model = load_model(args.gnn_type, args.model_path, data)
    errors, p_errors = test(model, data)

def parse_args():
    parser = argparse.ArgumentParser("Testing powerfloww GNN models")
    parser.add_argument("-g", "--gnn_type", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-d", "--data_path", required=True)
    args = parser.parse_args()
    return args

def test(model, data):
    loader = DataLoader(data)
    errors = []
    p_errors = []
    first = True
    for g in loader:
        out = model(g)
        error = th.sub(g.y, out)
        print("out: \n", out)
        print("y: \n", g.y)
        print("error: \n", error)
        p_error = th.div(error, g.y) * 100
        if first:
            print(error.shape)
            print(p_error.shape)
            first = False
        errors.append(error.detach().numpy().flatten())
        p_errors.append(p_error.detach().numpy().flatten())
    errors = np.concatenate(errors)
    p_errors = np.concatenate(p_errors)
    mask = np.isinf(p_errors)
    p_errors[mask] = 0

    print(p_errors)

    plt.hist(errors)
    plt.show()
    plt.hist(p_errors)
    plt.show()
    within = np.array((abs(p_errors) < 5)).sum() / len(p_errors)
    print("within", within)
    print("within 5%:", (abs(p_errors) < 5) / len(p_errors))
    return errors, p_errors

if __name__ == '__main__':
    main()