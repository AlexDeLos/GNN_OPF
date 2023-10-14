import os
import tqdm
import pandapower as pp
import pandapower.networks as pn
import argparse

def vis_graphs(dir):
    graph_path = f"{dir}/x"
    graph_paths = sorted(os.listdir(graph_path))
    data = []

    for g in tqdm.tqdm(enumerate(graph_paths)):
        graph = pp.from_json(f"{graph_path}/{g}")
        pn.simple_plot(graph)

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    args = parser.parse_args()
    vis_graphs(args.dir)