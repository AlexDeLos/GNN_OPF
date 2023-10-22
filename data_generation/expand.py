import argparse
from copy import deepcopy
import math
import numpy as np
import generate
import os
from pathlib import Path
import pandapower as pp
import random
import string
import tqdm
import time
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
    # if file is moved in another directory level relative to the root (currently in root/data_generation), this needs to be changed
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("-d", "--data_path", default=root_directory + "/Data")
    parser.add_argument("--dataset", choices=['train', 'val', 'test'], default='train')
    parser.add_argument("-s", "--save_dir", default=root_directory + "/Data")
    parser.add_argument("-n", "--num_networks", type=int, default=1)
    parser.add_argument("--from_case", choices=['case4gs', 'case5', 'case6ww', 'case9', 'case14', 'case24_ieee_rts', 'case30', 'case_ieee30', 'case39', 'case57', 'case89pegase', 'case118', 'case145', 'case_illinois200', 'case300', 'case1354pegase', 'case1888rte', 'case2848rte', 'case2869pegase', 'case3120sp', 'case6470rte', 'case6495rte', 'case6515rte', 'case9241', 'GBnetwork', 'GBreducednetwork', 'iceland'], default= None)
    args = parser.parse_args()
    print(args)
    return args


def expand(args):
    """
    Expands the given power network dataset by creating new networks with variations in load and generator power.

    Args:
        args: An object containing the following attributes:
            - data_path: The path to the directory containing the input dataset.
            - dataset: The name of the dataset to be expanded.
            - num_networks: The number of new networks to be generated for each input network.
            - save_dir: The path to the directory where the expanded dataset will be saved.

    Returns:
        n: The total number of new networks generated.
        time: The time taken to generate the new networks.
    """
    start = time.perf_counter()
    if args.from_case != None:
        print("Generating networks from scratch")
        network = generate.get_network(args.from_case)
        # Now run it on this graph
        for _ in range(args.num_networks):
            new_graph = pp.pandapowerNet(network.copy())
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
                new_graph.load.at[i,'q_mvar'] =np.abs(np.random.normal(new_graph.load.at[i,'q_mvar'], np.sqrt(abs(new_graph.load.at[i,'q_mvar'])))) # goes from -50 to about 300
            # Average value by which the load changes
            div = np.divide(np.array(new_p_mws),np.array(old_p_mws))
            # making sure that the average change is the same for the generators
            # this should be around one, but can vary.
            # The reason why this is needed is because we want the generators to be able to compensate for the load change.
            change = np.average(div)

            for i in new_graph.gen.index:
                new_graph.gen.at[i,'p_mw'] = new_graph.gen.at[i,'p_mw'] * change
                # new_graph.gen.at[i,'vm_pu'] = np.abs(np.random.normal(new_graph.gen.at[i,'vm_pu'], 1/100)) #goes from 0 - 1.1

            try:
                pp.runpp(new_graph, numba=False)
            except:
                print(f"Network not solvable trying a new one")
                continue
            # g_split = g.split("_")
            # network = g_split[0]
            subgraph_length = len(network.bus)
            uid = ''.join([random.choice(string.ascii_letters
                + string.digits) for _ in range(8)])  
            
            Path(f"{args.save_dir}/x").mkdir(parents=True, exist_ok=True)
            Path(f"{args.save_dir}/y").mkdir(parents=True, exist_ok=True)

            pp.to_json(new_graph, f"{args.save_dir}/x/{args.from_case}_{subgraph_length}_expanded_{uid}.json")
            new_graph.res_gen.to_csv(f"{args.save_dir}/y/{args.from_case}_{subgraph_length}_expanded_{uid}_gen.csv")
            new_graph.res_line.to_csv(f"{args.save_dir}/y/{args.from_case}_{subgraph_length}_expanded_{uid}_line.csv")
            new_graph.res_bus.to_csv(f"{args.save_dir}/y/{args.from_case}_{subgraph_length}_expanded_{uid}_bus.csv")
            n += 1
    else:
        p = f"{args.data_path}/{args.dataset}"
        print(f"Loading networks from {p}")
        graph_paths = sorted(os.listdir(f"{p}/x"))
        print(f"Num paths: {len(graph_paths)}")
        n = 0
        for g in tqdm.tqdm(graph_paths):
            graph = pp.from_json(f"{p}/x/{g}")
            for _ in range(args.num_networks):
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
                    new_graph.load.at[i,'q_mvar'] =np.abs(np.random.normal(new_graph.load.at[i,'q_mvar'], np.sqrt(abs(new_graph.load.at[i,'q_mvar'])))) # goes from -50 to about 300
                # Average value by which the load changes
                div = np.divide(np.array(new_p_mws),np.array(old_p_mws))
                # making sure that the average change is the same for the generators
                # this should be around one, but can vary.
                # The reason why this is needed is because we want the generators to be able to compensate for the load change.
                change = np.average(div)

                for i in new_graph.gen.index:
                    new_graph.gen.at[i,'p_mw'] = new_graph.gen.at[i,'p_mw'] * change
                    # new_graph.gen.at[i,'vm_pu'] = np.abs(np.random.normal(new_graph.gen.at[i,'vm_pu'], 1/100)) #goes from 0 - 1.1

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