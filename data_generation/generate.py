import pandapower as pp
import pandapower.plotting as ppl
import pandapower.networks as pn
import pandapower.toolbox as tb
import numpy as np
import os
import string
import random
import argparse
from collections import Counter
import time
from pathlib import Path
import subgraphs_methods

import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def generate():
    arguments = get_arguments()
    x, t = create_networks(arguments)
    print(f"{x} networks created in {t:0.2f} seconds")

def get_arguments():
    parser = argparse.ArgumentParser(
        prog="power network subgraph generator",
        description="Generates a specified number of subnetworks from a power network",
    )
    parser.add_argument("network", choices=['case4gs', 'case5', 'case6ww', 'case9', 'case14', 'case24_ieee_rts', 'case30', 'case_ieee30', 'case39', 'case57', 'case89pegase', 'case118', 'case145', 'case_illinois200', 'case300', 'case1354pegase', 'case1888rte', 'case2848rte', 'case2869pegase', 'case3120sp', 'case6470rte', 'case6495rte', 'case6515rte', 'case9241', 'GBnetwork', 'GBreducednetwork', 'iceland'])
    parser.add_argument("-n", "--num_subgraphs", type=int, default=10)

    # if file is moved in another directory level relative to the root (currently in root/data_generation), this needs to be changed
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("-s", "--save_dir", default=root_directory + "/Data") 

    parser.add_argument("--min_size", type=int, default=5)
    parser.add_argument("--max_size", type=int, default=30)
    parser.add_argument("--n_1", type=bool, default=False)
    parser.add_argument("--subgraphing_method", choices=['rnd_neighbor', 'bfs', 'rnd_walk'], default='rnd_neighbor')
    args = parser.parse_args()
    print(args)
    return args

def get_network(network_name):
    if network_name == 'case4gs':
        network = pn.case4gs()
    elif network_name == 'case5':
        network = pn.case5()
    elif network_name == 'case6ww':
        network = pn.case6ww()
    elif network_name == 'case14':
        network = pn.case14()
    elif network_name == 'case24_ieee_rts':
        network = pn.case24_ieee_rts()
    elif network_name == 'case30':
        network = pn.case30()
    elif network_name == 'case_ieee30':
        network = pn.case_ieee30()
    elif network_name == 'case39':
        network = pn.case39()
    elif network_name == 'case57':
        network = pn.case57()
    elif network_name == 'case89pegase':
        network = pn.case89pegase()
    elif network_name == 'case118':
        network = pn.case118()
    elif network_name == 'case145':
        network = pn.case145()
    elif network_name == 'case_illinois200':
        network = pn.case_illinois200()
    elif network_name == 'case300':
        network = pn.case300()
    elif network_name == 'case1354pegase': 
        network = pn.case1354pegase()
    elif network_name == 'case1888rte':
        network = pn.case1888rte()
    elif network_name == 'case2848rte':
        network = pn.case2848rte()
    elif network_name == 'case2869pegase': 
        network = pn.case2869pegase()
    elif network_name == 'case3120sp':
        network = pn.case3120sp()
    elif network_name == 'case6470rte':
        network = pn.case6470rte()
    elif network_name == 'case6495rte':
        network = pn.case6495rte()
    elif network_name == 'case6515rte':
        network = pn.case6515rte()
    elif network_name == 'case9241':
        network = pn.case9241()
    elif network_name == 'GBnetwork':
        network = pn.GBnetwork()
    elif network_name == 'GBreducednetwork':
        network = pn.GBreducednetwork()
    elif network_name == 'iceland':
        network = pn.iceland()
    return network


def get_subgraphing_method(method_name):
    if method_name == 'rnd_neighbor':
        return subgraphs_methods.random_neighbor_selection
    elif method_name == 'bfs':
        return subgraphs_methods.bfs_neighbor_selection
    elif method_name == 'rnd_walk':
        return subgraphs_methods.random_walk_neighbor_selection


def create_networks(arguments):
    start = time.perf_counter()
    full_net = get_network(arguments.network)

    # the external grid is plotted as a yellow rectangle, buses are blue dots, loads are black triangles with tip pointing down
    # transformers are red overlapping circles, generators are black circle with a small black propeller in the middle 
    # ppl.simple_plot(full_net, plot_loads=True, plot_gens=True, trafo_color="r", switch_color="g") 

    subgraphing_method = get_subgraphing_method(arguments.subgraphing_method)
    # A starting point is any bus that is connected to a generator to ensure that subgraphs contain at least one generator
    starting_points = full_net.gen.bus
    i = 0
    while i < arguments.num_subgraphs:
        print(f"generating network {i + 1}")
        
        subgraph_length = np.random.randint(arguments.min_size, min(arguments.max_size, len(full_net.bus)))
        initial_bus = starting_points[np.random.randint(0, len(starting_points))]
        
        if arguments.n_1:
            subgraph_busses = list(full_net.bus.index)
            downed_bus = np.random.randint(0, len(subgraph_busses))
            del subgraph_busses[downed_bus]
    
        else:
            subgraph_busses = subgraphing_method(full_net, initial_bus, subgraph_length)
        
        subgraph_net = tb.select_subnet(full_net, subgraph_busses)

        try:
            pp.runpp(subgraph_net)
            # ppl.simple_plot(subgraph_net, plot_loads=True, plot_gens=True, trafo_color="r", switch_color="g") 

        except:
            print(f"Network not solvable trying a new one")
            continue

        uid = ''.join([random.choice(string.ascii_letters
                + string.digits) for _ in range(8)])
        
        Path(f"{arguments.save_dir}/x").mkdir(parents=True, exist_ok=True)
        Path(f"{arguments.save_dir}/y").mkdir(parents=True, exist_ok=True)
        
        pp.to_json(subgraph_net, f"{arguments.save_dir}/x/{arguments.network}_{subgraph_length}_{arguments.subgraphing_method}_{uid}.json")
        subgraph_net.res_gen.to_csv(f"{arguments.save_dir}/y/{arguments.network}_{subgraph_length}_{arguments.subgraphing_method}_{uid}_gen.csv")
        subgraph_net.res_line.to_csv(f"{arguments.save_dir}/y/{arguments.network}_{subgraph_length}_{arguments.subgraphing_method}_{uid}_line.csv")
        subgraph_net.res_bus.to_csv(f"{arguments.save_dir}/y/{arguments.network}_{subgraph_length}_{arguments.subgraphing_method}_{uid}_bus.csv")

        i += 1
    end = time.perf_counter()
    return i, end - start


if __name__ == "__main__":
    generate()