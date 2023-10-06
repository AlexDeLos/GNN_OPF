import numpy as np


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

# other methods to try: k-hop neighborhood, Community Detection, random walk laplacian,graphSAINT partitioning, ...