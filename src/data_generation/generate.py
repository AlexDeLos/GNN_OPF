import pandapower as pp
import pandapower.plotting as ppl
import pandapower.networks as pn
import pandapower.toolbox as tb
import numpy as np
import string
import random
import argparse
from collections import Counter
import time
from pathlib import Path


def generate():
    arguments = get_arguments()
    x, t = create_networks(arguments)
    print(f"{x} networks created in {t:0.2f} seconds")

def get_arguments():
    parser = argparse.ArgumentParser(
        prog="power network subgraph generator",
        description="Generates a specified number of subnetworks from a power network",
    )
    parser.add_argument("network", choices=['case4gs', 'case5', 'case6ww', 'case9', 'case14', 'case24_ieee_rts', 'case30', 'case_ieee30', 'case33bw', 'case39', 'case57', 'case89pegase', 'case118', 'case145', 'case_illinois200', 'case300', 'case1354pegase', 'case1888rte', 'case2848rte', 'case2869pegase', 'case3120sp', 'case6470rte', 'case6495rte', 'case6515rte', 'case9241', 'GBnetwork', 'GBreducednetwork', 'iceland'])
    parser.add_argument("-n", "--num_subgraphs", type=int, default=10)
    parser.add_argument("-s", "--save_dir", default="./data")
    parser.add_argument("--min_size", type=int, default=5)
    parser.add_argument("--max_size", type=int, default=30)
    parser.add_argument("--n_1", type=bool, default=False)
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
    elif network_name == 'case33bw':
        network = pn.case33bw()
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

def create_networks(arguments):
    start = time.perf_counter()
    net = get_network(arguments.network)
    starting_points = net.gen.bus
    i = 0
    while i < arguments.num_subgraphs:
        print(f"generating network {i + 1}")
        length = np.random.randint(arguments.min_size, min(arguments.max_size, len(net.bus)))
        starting_point = starting_points[np.random.randint(0, len(starting_points))]
        busses = [starting_point]
        downed_bus = np.random.randint(0, length)
        while len(busses) < length:
            
            s = busses[np.random.randint(0, len(busses))]
            f = net.line.to_bus[np.where(np.array(net.line.from_bus) == s)[0]]
            t = net.line.from_bus[np.where(np.array(net.line.to_bus) == s)[0]]
            f_trafo = net.trafo.hv_bus[np.where(np.array(net.trafo.lv_bus) == s)[0]]
            t_trafo = net.trafo.lv_bus[np.where(np.array(net.trafo.hv_bus) == s)[0]]
            connected = np.concatenate((f, t, f_trafo, t_trafo))
            new_busses = np.setdiff1d(connected, busses)
            if len(new_busses) == 0:
                continue
            busses.append(new_busses[np.random.randint(0, len(new_busses))])
        
        if arguments.n_1:
            busses[downed_bus] = 0
        new_net = tb.select_subnet(net, busses)

        try:
            pp.runpp(new_net)
        except:
            print(f"Network not solvable trying a new one")
            continue

        uid = ''.join([random.choice(string.ascii_letters
                + string.digits) for _ in range(8)])
        
        Path(f"{arguments.save_dir}/x").mkdir(parents=True, exist_ok=True)
        Path(f"{arguments.save_dir}/y").mkdir(parents=True, exist_ok=True)
        
        pp.to_json(new_net, f"{arguments.save_dir}/x/{arguments.network}_{length}_{uid}.json")
        new_net.res_gen.to_csv(f"{arguments.save_dir}/y/{arguments.network}_{length}_{uid}_gen.csv")
        new_net.res_line.to_csv(f"{arguments.save_dir}/y/{arguments.network}_{length}_{uid}_line.csv")
        new_net.res_bus.to_csv(f"{arguments.save_dir}/y/{arguments.network}_{length}_{uid}_bus.csv")

        i += 1
    end = time.perf_counter()
    return i, end - start


if __name__ == "__main__":
    generate()
