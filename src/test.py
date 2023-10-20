import torch as th
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from utils import load_data, normalize_data
from utils_hetero import normalize_data_hetero


from utils import load_data_helper, load_model, read_from_pkl, write_to_pkl, load_model_hetero  


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
        model = load_model_hetero(args.gnn_type, args.model_path, data, args)
    else:
        model = load_model(args.gnn_type, args.model_path, data, args)
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
        # if first:
        #     print("Y")
        #     print(g.y)
        #     print("Pred")
        #     print(out)
            # quit()
            # first = False
            # print(th.cat([g.y, out], dim=1))
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

    # plt.hist(errors)
    # plt.show()
    # plt.hist(p_errors)
    # plt.show()
    print("within 5%", np.sum(abs(p_errors) < 5, axis=0) / len(p_errors))
    print("within 10%", np.sum(abs(p_errors) < 10, axis=0) / len(p_errors))
    print("within 15%", np.sum(abs(p_errors) < 15, axis=0) / len(p_errors))
    print("within 25%", np.sum(abs(p_errors) < 25, axis=0) / len(p_errors))
    print("within 50%", np.sum(abs(p_errors) < 50, axis=0) / len(p_errors))

    return errors, p_errors

def test_hetero(model, data):
    loader = DataLoader(data)
    load_errors = []
    gen_errors = []
    load_gen_errors = []
    first = True
    for g in loader:
        out = model(g.x_dict, g.edge_index_dict)
        if first:
            for node_type, y in g.y_dict.items():
                print('in test', node_type)
                print(th.cat((out[node_type], y), axis=1))
            quit()
        error = th.abs(th.sub(g.y, out))
        p_error = th.div(error, g.y) * 100
        errors.append(error.detach().numpy())
        p_errors.append(p_error.detach().numpy())

    errors = np.concatenate(errors)
    errors = errors.reshape((-1, 2))

    p_errors = np.concatenate(p_errors)
    p_errors = p_errors.reshape((-1, 2))
    print(np.shape(p_errors))

    mask = np.isinf(p_errors)
    p_errors[mask] = 0

    plt.hist(errors)
    plt.show()
    plt.hist(p_errors)
    plt.show()

    print("within 5%", np.sum(abs(p_errors) < 5, axis=0) / len(p_errors))
    print("within 10%", np.sum(abs(p_errors) < 10, axis=0) / len(p_errors))
    print("within 15%", np.sum(abs(p_errors) < 15, axis=0) / len(p_errors))
    print("within 25%", np.sum(abs(p_errors) < 25, axis=0) / len(p_errors))
    print("within 50%", np.sum(abs(p_errors) < 50, axis=0) / len(p_errors))

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