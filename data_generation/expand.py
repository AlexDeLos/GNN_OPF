import argparse
from copy import deepcopy
import math
import numpy as np
import os
from pathlib import Path
import pandapower as pp
import random
import string
import tqdm
import time
import generate
import warnings
from generate import modify_network_values
# Suppress all warnings
warnings.filterwarnings("ignore")


def main():
    args = get_args()
    n, t = expand(args)
    print(f"{n} new networks expanded in {t:0.2f} seconds")


def get_args():
    parser = argparse.ArgumentParser(
        prog="power network subgraph generator",
        description="Generates a specified number of subnetworks from a power network"
    )
    # if file is moved in another directory level relative to the root (currently in root/data_generation), this needs to be changed
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("-d", "--data_path", default=root_directory + "/Data")
    parser.add_argument("--dataset", choices=['train', 'val', 'test'], default='train')
    parser.add_argument("-s", "--save_dir", default=root_directory + "/Data")
    parser.add_argument("-n", "--num_networks", type=int, default=1)
    parser.add_argument("--down_lines", type=int, default=0)
    parser.add_argument("--no_leakage", action="store_true", default=False)
    parser.add_argument("--from_case", choices=['case4gs', 'case5', 'case6ww', 'case9', 'case14', 'case24_ieee_rts', 'case30', 'case_ieee30', 'case39', 'case57', 'case89pegase', 'case118', 'case145', 'case_illinois200', 'case300', 'case1354pegase', 'case1888rte', 'case2848rte', 'case2869pegase', 'case3120sp', 'case6470rte', 'case6495rte', 'case6515rte', 'case9241', 'GBnetwork', 'GBreducednetwork', 'iceland'], default= None)
    args = parser.parse_args()
    print(args)
    return args

def expand_helper(args, graph, name):
    num_generated_graphs = 0
    trials = 0
    n = 0
    while num_generated_graphs < args.num_networks:
        if trials == 30: # avoid infinite loop, stop after 30 trials
            print(f"Could not generate a new network for {name} in 30 trials, skipping...")
            break
        new_graph = pp.pandapowerNet(graph.copy())
        old_p_mws= []
        new_p_mws = []
        #basis from all the variation in the network
        #decided for a uniform distribution
        ratio_increase = np.random.uniform(0.8,1.2)
        # iterate over the indices of the loads
        for i in new_graph.load.index:
            individual_variation = np.random.normal(0,0.05)
            if new_graph.load.at[i,'p_mw']==0:
                # safety to aviod dividing by 0
                new_graph.load.at[i,'p_mw'] =  0.00000000001
            old_p_mws.append(new_graph.load.at[i,'p_mw'])
            new_graph.load.at[i,'p_mw'] = new_graph.load.at[i,'p_mw'] * (ratio_increase + individual_variation)
            new_p_mws.append(new_graph.load.at[i,'p_mw'])
            #same correlation of the P_mw
            new_graph.load.at[i,'q_mvar'] = new_graph.load.at[i,'q_mvar'] * (ratio_increase + individual_variation)
        # Average value by which the load changes
        div = np.divide(np.array(new_p_mws),np.array(old_p_mws))
        # making sure that the average change is the same for the generators
        # this should be around one, but can vary.
        # The reason why this is needed is because we want the generators to be able to compensate for the load change.
        change = np.average(div)

        for i in new_graph.gen.index:
            new_graph.gen.at[i,'p_mw'] = new_graph.gen.at[i,'p_mw'] * change
    
        i = 0
        changed_lines = []
        while i != args.down_lines:
            random_line = random.choice(new_graph.line.index)
            if new_graph.line.at[random_line,'in_service'] == True:
                new_graph.line.at[random_line,'in_service'] = False
                changed_lines.append(random_line)
                i += 1
        
            

        try:
            if args.no_leakage:
                print("no leakage")
                new_graph = modify_network_values(new_graph)

            pp.runpp(new_graph, numba=False)
            num_generated_graphs += 1
            trials = 0
        except:
            trials += 1
            continue

        subgraph_length = len(new_graph.bus)
        uid = ''.join([random.choice(string.ascii_letters
            + string.digits) for _ in range(8)])  
        
        Path(f"{args.save_dir}/x").mkdir(parents=True, exist_ok=True)
        Path(f"{args.save_dir}/y").mkdir(parents=True, exist_ok=True)

        pp.to_json(new_graph, f"{args.save_dir}/x/{name}_{subgraph_length}_expanded_{uid}.json")
        new_graph.res_gen.to_csv(f"{args.save_dir}/y/{name}_{subgraph_length}_expanded_{uid}_gen.csv")
        new_graph.res_line.to_csv(f"{args.save_dir}/y/{name}_{subgraph_length}_expanded_{uid}_line.csv")
        new_graph.res_bus.to_csv(f"{args.save_dir}/y/{name}_{subgraph_length}_expanded_{uid}_bus.csv")
        for line in changed_lines:
            new_graph.line.at[line,'in_service'] = True
        print(num_generated_graphs)
    return num_generated_graphs

def expand(args):
    
    
    start = time.perf_counter()
    if args.from_case != None:
        print("Generating networks from scratch")
        graph = generate.get_network(args.from_case)
        n=0
        while n<args.num_networks:
            n += expand_helper(args, graph.copy(), args.from_case) 
            args.num_networks -= n
    else:
        p = f"{args.data_path}/{args.dataset}"
        print(f"Loading networks from {p}")
        graph_paths = sorted(os.listdir(f"{p}/x"))
        print(f"Num paths: {len(graph_paths)}")
        n = 0
        for g in tqdm.tqdm(graph_paths):
            graph = pp.from_json(f"{p}/x/{g}")
            n += expand_helper(args, graph, g)
    end = time.perf_counter()
    return n, end - start

if __name__ == "__main__":
    main()

