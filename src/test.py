import torch as th
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader


from utils import load_data_helper, load_model, read_from_pkl, write_to_pkl

def main():
    args = parse_args()
    try:
        data = read_from_pkl(f"{args.data_path}/pickled.pkl")
    except:
        data = load_data_helper(f"{args.data_path}/pickled.pkl")
        write_to_pkl(data, f"{args.data_path}/pickled.pkl")
    model = load_model(args.gnn_type, args.model_path, data)
    model.eval()
    new_test(model, data)

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
        # print(g.y)
        # print(out)
        # quit()
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

    # plt.hist(errors)
    # plt.show()
    # plt.hist(p_errors)
    # plt.show()

    num_correct = np.sum(abs(p_errors) < 5, axis=0)
    
    print("within", num_correct / len(p_errors))
    return errors, p_errors

def new_test(model, data):
    loader = DataLoader(data)
    load_errors = []
    p_load_errors = []

    gen_errors = []
    p_gen_errors = []

    ext_errors = []
    p_ext_errors = []

    none_errors = []
    p_none_errors = []

    first = True
    for g in loader:
        out = model(g)
        error = th.sub(g.y, out)
        p_error = th.div(error, g.y) * 100
        load_indices = (g.x[:, 0] == 1).nonzero()
        gen_indices = (g.x[:, 1] == 1).nonzero()
        ext_indices = (g.x[:, 2] == 1).nonzero()
        none_indices = (g.x[:, 3] == 1).nonzero()

        if first:
            print(len(load_indices))
            print(len(gen_indices))
            print(len(ext_indices))
            print(len(none_indices))
            first = False

        load_errors.append(error[load_indices].detach().numpy())
        gen_errors.append(error[gen_indices].detach().numpy())
        ext_errors.append(error[ext_indices].detach().numpy())
        none_errors.append(error[none_indices].detach().numpy())

        p_load_errors.append(p_error[load_indices].detach().numpy())
        p_gen_errors.append(p_error[gen_indices].detach().numpy())
        p_ext_errors.append(p_error[ext_indices].detach().numpy())
        p_none_errors.append(p_error[none_indices].detach().numpy())


    process_errors(p_load_errors)
    process_errors(p_gen_errors)
    process_errors(p_ext_errors)
    process_errors(p_none_errors)

def process_errors(errors):
        errors = np.concatenate(errors)
        errors = errors.reshape((-1, 2))
        print(len(errors))
        mask = np.isinf(errors)
        errors[mask] = 0
        num_correct_5 = np.sum(abs(errors) < 5, axis=0)
        num_correct_10 = np.sum(abs(errors) < 10, axis=0)
        num_correct_15 = np.sum(abs(errors) < 15, axis=0)
        num_correct_25 = np.sum(abs(errors) < 25, axis=0)
        num_correct_50 = np.sum(abs(errors) < 50, axis=0)

        print("within", num_correct_5 / len(errors))
        print("within", num_correct_10 / len(errors))
        print("within", num_correct_15 / len(errors))
        print("within", num_correct_25 / len(errors))
        print("within", num_correct_50 / len(errors))

if __name__ == '__main__':
    main()