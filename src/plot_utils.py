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
    # change directory to root of project
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.exists("plots"):
        os.mkdir("plots")

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    model_name = "distance_plot" + "_" + model.class_name + "_" + str(timestamp)
    plt.savefig(f"plots/{model_name}.png", format="png")
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
        #if the p_mw_gen is > 0 then it is a generator
        vm_pu = node[2]
        p_mw_gen = node[1]
        if vm_pu > 0 and p_mw_gen > 0:
            distances.append(BFS(data, node_index))

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


def BFS(graph, source):
    """
    Perform a breadth-first search on a graph starting from a given source node.

    Args:
        graph (Graph): The graph to search.
        source (int): The index of the source node.

    Returns:
        list: A list of lists, where each inner list contains the indices of nodes at a given depth
              from the source node. The outer list contains all the inner lists in order of increasing depth.
    """
    # Mark all the vertices as not visited
    visited = [False] * graph.edge_index.shape[1]

    # Create a queue for BFS
    queue = []

    
    bfs_neighbors = [[]]
    bfs_neighbors[0].append(source)
    depth = -1

    # Mark the source node as
    # visited and enqueue it
    queue.append(source)
    visited[source] = True

    while queue:

        # Dequeue a vertex from
        # queue and print it
        s = queue.pop(0)
        
        depth +=1
        if depth != 0:
            bfs_neighbors.append([])

        # Get all adjacent vertices of the
        # dequeued vertex s.
        # If an adjacent has not been visited,
        # then mark it visited and enqueue it
        for node_connected in get_neighbors(graph, s):
            if visited[node_connected] == False:
                queue.append(node_connected)
                visited[node_connected] = True
                bfs_neighbors[depth].append(node_connected)
    return bfs_neighbors