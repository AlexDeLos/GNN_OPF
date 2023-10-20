import numba
import numpy as np
import networkx as nx
import pandapower as pp
import community


def random_neighbor_selection(full_net, initial_bus, subgraph_length):
    subgraph_busses = [initial_bus]
    while len(subgraph_busses) < subgraph_length:
        # Pick random bus in the current subgraph
        s = subgraph_busses[np.random.randint(0, len(subgraph_busses))]
        # Get all busses directly connected to the picked bus s (from/to)
        f = full_net.line.to_bus[np.where(np.array(full_net.line.from_bus) == s)[0]]
        t = full_net.line.from_bus[np.where(np.array(full_net.line.to_bus) == s)[0]]
        # Get all busses connected to the picked bus s via a transformer (trafo) either high voltage (hv) or low voltage (lv)
        f_trafo = full_net.trafo.hv_bus[np.where(np.array(full_net.trafo.lv_bus) == s)[0]]
        t_trafo = full_net.trafo.lv_bus[np.where(np.array(full_net.trafo.hv_bus) == s)[0]]
        connected = np.concatenate((f, t, f_trafo, t_trafo))
        # Remove all busses that are already in the subgraph
        new_busses = np.setdiff1d(connected, subgraph_busses)
        if len(new_busses) == 0:
            continue
        # Pick a random bus from the remaining busses in new_busses
        subgraph_busses.append(new_busses[np.random.randint(0, len(new_busses))])
    
    return subgraph_busses


def bfs_neighbor_selection(full_net, starting_bus, subgraph_length):
    # Initialize a queue for BFS and a list to store visited buses
    queue = [starting_bus]
    visited = [starting_bus]

    while len(visited) < subgraph_length and queue:
        # Dequeue a bus from the front of the queue
        current_bus = queue.pop(0)

        # Get all busses directly connected to the current bus (from/to)
        f = full_net.line.to_bus[np.where(np.array(full_net.line.from_bus) == current_bus)[0]]
        t = full_net.line.from_bus[np.where(np.array(full_net.line.to_bus) == current_bus)[0]]

        # Get all busses connected to the current bus via a transformer (trafo) either high voltage (hv) or low voltage (lv)
        f_trafo = full_net.trafo.hv_bus[np.where(np.array(full_net.trafo.lv_bus) == current_bus)[0]]
        t_trafo = full_net.trafo.lv_bus[np.where(np.array(full_net.trafo.hv_bus) == current_bus)[0]]

        connected = np.concatenate((f, t, f_trafo, t_trafo))
        
        for neighbor in connected:
        # If the neighbor has not been visited, enqueue it and mark it as visited
            if neighbor not in visited and len(visited) < subgraph_length:
                queue.append(neighbor)
                visited.append(neighbor)

    return visited


def random_walk_neighbor_selection(full_net, starting_bus, subgraph_length):
    restart_prob=0.1
    previous_step_bus = starting_bus
    current_bus = starting_bus
    subgraph_busses = [starting_bus]
    stuck_iteration = 0

    while len(subgraph_busses) < subgraph_length:
        # Check if we should restart the random walk with a certain probability
        if np.random.rand() < restart_prob:
            current_bus = starting_bus

        # Get all busses directly connected to the current bus (from/to)
        f = full_net.line.to_bus[np.where(np.array(full_net.line.from_bus) == current_bus)[0]]
        t = full_net.line.from_bus[np.where(np.array(full_net.line.to_bus) == current_bus)[0]]

        # Get all busses connected to the current bus via a transformer (trafo) either high voltage (hv) or low voltage (lv)
        f_trafo = full_net.trafo.hv_bus[np.where(np.array(full_net.trafo.lv_bus) == current_bus)[0]]
        t_trafo = full_net.trafo.lv_bus[np.where(np.array(full_net.trafo.hv_bus) == current_bus)[0]]

        connected = np.concatenate((f, t, f_trafo, t_trafo))

        # Remove all busses that are already in the subgraph
        new_busses = np.setdiff1d(connected, subgraph_busses)

        
        if len(new_busses) == 0:
            stuck_iteration += 1
            if stuck_iteration > 20:
                # pick a random bus from the subgraph
                # random walk with random backtracking to avoid getting stuck in dead ends
                current_bus = subgraph_busses[np.random.randint(0, len(subgraph_busses))]
                stuck_iteration = 0
                continue
            else:
                # pick the previous bus
                current_bus = previous_step_bus
                continue

        # Pick a random bus from the remaining busses in new_busses
        next_bus = new_busses[np.random.randint(0, len(new_busses))]

        # Move to the next bus in the random walk
        previous_step_bus = current_bus
        current_bus = next_bus
        subgraph_busses.append(current_bus)
        stuck_iteration = 0

    return subgraph_busses

# Creates a network with the same topology as the full network, but with random variations for the loads and generators values
def number_changes(full_net):
    test_net = pp.pandapowerNet(full_net.copy())
    #TODO
    # We vary every value based on how big they are around their point.
    # values should vary so that they are not too far from their original value.
    # assume 80% correlation between every node.
    # Power should increase as much as the load increases.
    old = []
    new = []
    #basis from all the variation in the network
    #decided for a uniform distribution
    ratio_increase = np.random.uniform(0.9,1.1)
    for i in range(len(test_net.load)):
        individual_variation = np.random.normal(0,0.05)
        if test_net.load.at[i,'p_mw']==0:
            test_net.load.at[i,'p_mw'] =  0.00000000001
        old.append(test_net.load.at[i,'p_mw'])
        test_net.load.at[i,'p_mw'] = test_net.load.at[i,'p_mw'] * (ratio_increase + individual_variation)
        new.append(test_net.load.at[i,'p_mw'])
        #same correlation of the P_mw
        test_net.load.at[i,'q_mvar'] =np.abs(np.random.normal(test_net.load.at[i,'q_mvar'], np.sqrt(abs(test_net.load.at[i,'q_mvar'])))) # goes from -50 to about 300
    # Average value by which the load changes
    div = np.divide(np.array(new),np.array(old))
    # np.nan_to_num(div)
    change = np.average(div)

    for i in range(len(test_net.gen)):
        individual_variation = np.random.normal(0,0.05)
        test_net.gen.at[i,'p_mw'] = test_net.gen.at[i,'p_mw'] * change #(ratio_increase + individual_variation) #np.abs(np.random.normal(test_net.gen.at[i,'p_mw'], np.sqrt(abs(test_net.gen.at[i,'p_mw'])/100))) #goes from 0 - 1000

        # test_net.gen.at[i,'vm_pu'] = np.abs(np.random.normal(test_net.gen.at[i,'vm_pu'], 1/100)) #goes from 0 - 1.1
    

    return test_net


def partition_graph(full_net, min_partition_size=5):
        
    # Create a graph from the network data
    G = nx.Graph()
    G.add_nodes_from(full_net.bus.index)

    for _, line in full_net.line.iterrows():
        G.add_edge(line.from_bus, line.to_bus)

    # https://python-louvain.readthedocs.io/en/latest/api.html
    # https://ieeexplore.ieee.org/document/8245596
    partition = community.best_partition(G)

    # Create a dictionary to map cluster IDs to lists of bus IDs
    clusters = {}
    for node, cluster_id in partition.items():
        if cluster_id in clusters:
            clusters[cluster_id].append(node)
        else:
            clusters[cluster_id] = [node]

    # Convert the dictionary values to a list of lists
    all_partitions_busses = list(clusters.values())

    # Remove all partitions that are smaller than the min_partition_size
    all_partitions_busses = [partition_busses for partition_busses in all_partitions_busses if len(partition_busses) >= min_partition_size]
    
    return all_partitions_busses
