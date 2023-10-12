import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import torch as th
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_losses(losses, val_losses, model_name):
    """
    Plots the training and validation losses for a given model.

    Args:
        losses (list): List of training losses.
        val_losses (list): List of validation losses.
        model_name (str): Name of the model.

    Returns:
        None
    """
    epochs = np.arange(len(losses))

    plt.subplot(1, 2, 1)
    plt.title(f"{model_name} - Power Flow Training Learning Curve")
    plt.plot(epochs, losses, label="Training Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.subplot(1, 2, 2)
    plt.title(f"{model_name} - Power Flow Validation Learning Curve")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.tight_layout()
    plt.show()


def distance_plot(model, batch):
    """
    Plots the error with distance from the generator for a given model and batch.

    Args:
        model (torch.nn.Module): The model to use for generating predictions.
        batch (torch_geometric.data.Batch): The batch of data to use for plotting.

    Returns:
        None
    """
    out = model(batch)
    distance_loss,len = get_distance_loss(out,batch.y,batch)
    plt.bar(list(range(0,len)), distance_loss, color ='maroon')
    plt.title("Error with distance from the generator")
    plt.ylabel("Error")
    plt.xticks(range(0,len))
    plt.xlabel("Nodes away from the generator the node was located")
    
    # if file is moved in another directory level relative to the root (currently in root/src), this needs to be changed
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plot_directory = root_directory + "/plots"
    if not os.path.exists(plot_directory):
        os.mkdir(plot_directory)

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    model_name = "distance_plot" + "_" + model.class_name + "_" + str(timestamp)
    plt.savefig(f"{plot_directory}/{model_name}.png", format="png")
    plt.show()



def get_distance_loss(out, labels, data):
    """
    Calculates the distance loss between the predicted output and the ground truth labels.

    Args:
        out (torch.Tensor): The predicted output tensor.
        labels (torch.Tensor): The ground truth label tensor.
        data (DataGenerator): The data generator object.

    Returns:
        Tuple[List[float], int]: A tuple containing the list of distance losses and the length of the list.
    """
    res = [0]
    normalization_vector = [0]
    distances = get_distance_from_generator(data)
    for i, dis in enumerate(distances):
        if dis != -1:
            if dis > len(res)-1:
                res = res + [0]*(dis-len(res) +1)
                normalization_vector = normalization_vector + [0]*(dis-len(normalization_vector) +1)
            normalization_vector[dis] += 1
            res[dis] += th.sum(th.abs(out[i]-labels[i])).item()
    for i in range(len(res)):
        if normalization_vector[i] != 0:
            res[i] = res[i]/normalization_vector[i]
    return res, len(res)



def get_distance_from_generator(data):
    """
    Calculates the distance of each node in the graph from the nearest generator node.

    Args:
        data (torch_geometric.data.Data): The graph data.

    Returns:
        list: A list of distances of each node from the nearest generator node. Each element in the list is a list at index i is i away from a generator
    """
    distances = []
    for node_index, node in enumerate(data.x):
        #if the p_mw_gen is > 0 and vm_pu > 0 then it is a generator
        vm_pu = node[2]
        p_mw_gen = node[1]
        if vm_pu > 0 and p_mw_gen > 0:
            dis = [[]]
            dfs(set(), data, node_index, 0, dis)
            distances.append(dis)

    result = [-1]*len(data.x)
    for el in range(0, len(result)):
        max_distance = 999999999
        for distance in distances:
            for node_index in distance:
                if el in node_index and max_distance > distance.index(node_index):
                    max_distance = distance.index(node_index)
        if max_distance != 999999999:
            result[el] = max_distance
    return result


def get_neighbors(data, node):
    """
    Given a PyTorch Geometric `Data` object and a node index, returns a set of the indices of all neighboring nodes.
    
    Args:
        data (torch_geometric.data.Data): A PyTorch Geometric `Data` object representing a graph.
        node (int): The index of the node whose neighbors are to be found.
    
    Returns:
        set: A set of integers representing the indices of all neighboring nodes.
    """
    neighbors = set()
    edges = data.edge_index

    #assume they are ordered
    for edge_idex,node_at_other_end in enumerate(edges[0,:]):
        if node_at_other_end.item() == node:
            neighbors.add(edges[1,edge_idex].item())
    
    return neighbors


def dfs(visited, graph, node, depth, ret_array):
    """
    Perform a depth-first search on a graph starting from a given source node.

    Args:
        visited (set): A set of visited nodes.
        graph (Graph): The graph to search.
        node (int): The index of the current node.
        depth (int): The current depth of the search.
        ret_array (list): A list of lists, where each inner list contains the indices of nodes at a given depth
                          from the source node. The outer list contains all the inner lists in order of increasing depth.

    Returns:
        None
    """
    this_path_is_shorter = True
    if node in visited:
        for dep in range(0,depth):
            for i in range(len(ret_array[dep])):
                if ret_array[dep][i] == node:
                    this_path_is_shorter = False
                    break

    if node not in visited or this_path_is_shorter:
        if len(ret_array) <= depth:
            ret_array.append([])
        ret_array[depth].append(node)
        visited.add(node)
        for node_connected in get_neighbors(graph, node):
            dfs(visited, graph, node_connected, depth + 1, ret_array)

