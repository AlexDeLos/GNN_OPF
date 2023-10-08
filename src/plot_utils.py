import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import torch as th

def plot_losses(losses, val_losses, model_name):
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
    plt.savefig(f"plots/{model_name}.png")
    plt.show()


def get_distance_loss(out,labels,data):
    res = [0]
    norm = [0]
    distances = get_distance_from_generator(data)
    for i, dis in enumerate(distances):
        if dis != -1:
            if dis > len(res)-1:
                res = res + [0]*(dis-len(res) +1)
                norm = norm + [0]*(dis-len(norm) +1)
            norm[dis] += 1
            res[dis] += th.sum(th.abs(out[i]-labels[i])).item()
    for i in range(len(res)):
        if norm[i] != 0:
            res[i] = res[i]/norm[i]
    return res, len(res)


def MES_loss(cur,out,label):
    return th.add(cur+th.abs(out-label))


def get_distance_from_generator(data):
    distances = []
    for i, node in enumerate(data.x):
        #if the p_mw_gen is > 0 then it is a generator
        vm_pu = node[2]
        if vm_pu > 0:
            distances.append(BFS(data, i))

    result = [-1]*len(data.x)
    for el in range(0, len(result)):
        max_distance = 999999999
        for distance in distances:
            for i in distance:
                if el in i and max_distance > distance.index(i):
                    max_distance = distance.index(i)
        if max_distance != 999999999:
            result[el] = max_distance
    return result


def get_neighbors(data, node):
    neighbors = set()
    edges = data.edge_index

    #assume they are ordered
    #broken:
    for i,node1 in enumerate(edges[0,:]):
        if node1.item() == node:
            neighbors.add(edges[1,i].item())
    
    return neighbors


def BFS(graph, source):

    # Mark all the vertices as not visited
    visited = [False] * graph.edge_index.shape[1]

    # Create a queue for BFS
    queue = []

    
    array = [[]]
    array[0].append(source)
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
            array.append([])

        # Get all adjacent vertices of the
        # dequeued vertex s.
        # If an adjacent has not been visited,
        # then mark it visited and enqueue it
        for i in get_neighbors(graph, s):
            if visited[i] == False:
                queue.append(i)
                visited[i] = True
                array[depth].append(i)
    return array