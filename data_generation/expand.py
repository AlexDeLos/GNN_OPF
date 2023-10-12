import pandapower as pp
from copy import deepcopy
import numpy as np
import math
import tqdm
import os
import argparse
import time
import random
import string
from pathlib import Path

import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    args = get_args()
    n, t = expand(args)
    print(f"{n} new networks expanded in {t:0.2f} seconds")

def get_args():
    parser = argparse.ArgumentParser(
        prog="power network subgraph generator",
        description="Generates a specified number of subnetworks from a power network"
    )
    # if file is moved in another directory (currently in root/data_generation), this needs to be changed
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("-d", "--data_path", default=root_directory + "/Data/train")
    parser.add_argument("-s", "--save_dir", default=root_directory + "/Data/expand")
    args = parser.parse_args()
    print(args)
    return args

def expand(args):
    start = time.perf_counter()
    p = args.data_path
    graph_paths = sorted(os.listdir(f"{p}/x"))
    print(f"Num paths: {len(graph_paths)}")
    n = 0
    for g in tqdm.tqdm(graph_paths):
        graph = pp.from_json(f"{p}/x/{g}")
        for _ in range(int(len(graph.bus) / 4)):
            new_graph = deepcopy(graph)
            for i in graph.gen.index:
                new_graph.gen.loc[i, 'p_mw'] = np.random.normal(graph.gen.loc[i, 'p_mw'], math.sqrt(abs(graph.gen.loc[i, 'p_mw'])))
            for i in graph.load.index:
                new_graph.load.loc[i, 'p_mw'] = np.random.normal(graph.load.loc[i, 'p_mw'], math.sqrt(abs(graph.load.loc[i, 'p_mw'])))
                new_graph.load.loc[i, 'q_mvar'] = np.random.normal(graph.load.loc[i, 'q_mvar'], math.sqrt(abs(graph.load.loc[i, 'q_mvar'])))

            try:
                pp.runpp(new_graph, numba=False)
            except:
                print(f"Network not solvable trying a new one")
                continue
            g_split = g.split("_")
            network = g_split[0]
            subgraph_length = g_split[1]
            uid = ''.join([random.choice(string.ascii_letters
                + string.digits) for _ in range(8)])  
            
            Path(f"{args.save_dir}/x").mkdir(parents=True, exist_ok=True)
            Path(f"{args.save_dir}/y").mkdir(parents=True, exist_ok=True)

            pp.to_json(new_graph, f"{args.save_dir}/x/{network}_{subgraph_length}_expanded_{uid}.json")
            new_graph.res_gen.to_csv(f"{args.save_dir}/y/{network}_{subgraph_length}_expanded_{uid}_gen.csv")
            new_graph.res_line.to_csv(f"{args.save_dir}/y/{network}_{subgraph_length}_expanded_{uid}_line.csv")
            new_graph.res_bus.to_csv(f"{args.save_dir}/y/{network}_{subgraph_length}_expanded_{uid}_bus.csv")
            n += 1
        # quit()
    end = time.perf_counter()
    return n, end - start

if __name__ == "__main__":
    main()