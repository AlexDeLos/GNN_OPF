import torch as th
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from utils import load_data


from utils import load_data_helper, load_model

def main():
    args = parse_args()
    #Call it with the same path for all three arguments in order to use only the train data
    train, val, data = load_data(args.data_path,args.data_path,args.data_path)
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
    for g in loader:
        out = model(g)
        error = th.sub(g.y, out)
        p_error = th.div(error, g.y) * 100
        errors.append(error.detach().numpy())
        p_errors.append(p_error.detach().numpy())

    errors = np.concatenate(errors)
    errors = errors.reshape((-1, 4))

    p_errors = np.concatenate(p_errors)
    p_errors = p_errors.reshape((-1, 4))

    mask = np.isinf(p_errors)
    p_errors[mask] = 0

    plt.hist(errors)
    plt.show()
    plt.hist(p_errors)
    plt.show()

    num_correct = np.sum(abs(p_errors) < 5, axis=0)
    
    print("within", num_correct / len(p_errors))
    return errors, p_errors

if __name__ == '__main__':
    main()