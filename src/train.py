from torch_geometric.loader import DataLoader as pyg_DataLoader
import tqdm
from queue import Queue
import torch as th
import utils
import matplotlib.pyplot as plt 
from utils import get_gnn, get_optim, get_criterion


def train_model(arguments, train, val, test):
    input_dim = train[0].x.shape[1]
    edge_attr_dim = train[0].edge_attr.shape # why not [1]
    output_dim = train[0].y.shape[1]

    print(f"Input shape: {input_dim}\nOutput shape: {output_dim}")

    batch_size = arguments.batch_size
    train_dataloader = pyg_DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = pyg_DataLoader(val, batch_size=batch_size, shuffle=False)
    gnn_class = get_gnn(arguments.gnn)
    gnn = gnn_class(input_dim, output_dim, edge_attr_dim)
    print(f"GNN: \n{gnn}")

    optimizer_class = get_optim(arguments.optimizer)
    optimizer = optimizer_class(gnn.parameters(), lr=arguments.learning_rate)
    criterion = get_criterion(arguments.criterion)

    losses = []
    val_losses = []
    for epoch in tqdm.tqdm(range(arguments.n_epochs)): #args epochs
        epoch_loss = 0.0
        epoch_val_loss = 0.0
        gnn.train()
        for batch in train_dataloader:
            epoch_loss += train_batch(data=batch, model=gnn, optimizer=optimizer, criterion=criterion)
        gnn.eval()
        for batch in val_dataloader:
            # distance_plot(gnn, batch, True)
            epoch_val_loss += evaluate_batch(data=batch, model=gnn, criterion=criterion)

        avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
        avg_epoch_val_loss = epoch_val_loss.item() / len(val_dataloader)

        losses.append(avg_epoch_loss)
        val_losses.append(avg_epoch_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.6f}, val_loss: {avg_epoch_val_loss:.6f}')
        if epoch == arguments.n_epochs-1 and arguments.plot_node_error:
            distance_plot(gnn, batch,True)
        #Early stopping
        try:  
            if val_losses[-1]>=val_losses[-2]:
                early_stop += 1
                if early_stop == arguments.patience:
                    print("Early stopping! Epoch:", epoch)
                    distance_plot(gnn, batch)
                    break
            else:
                early_stop = 0
        except:
            early_stop = 0
    
    return gnn, losses, val_losses

def distance_plot(model, batch, show = False):
    out = model(batch)
    distance_loss,len = get_distance_loss(out,batch.y,batch)
    if(show):
        plt.bar(list(range(0,len)), distance_loss, color ='maroon')
        plt.title("Error with distance from the generator")
        plt.ylabel("Error")
        plt.xticks(range(0,len))
        plt.xlabel("Nodes away from the generator the node was located")
        plt.show()

def train_batch(data, model, optimizer, criterion, device='cpu'):
    model.to(device)
    optimizer.zero_grad()
    out = model(data)
    # each element i in this array represents the distance of the node i from the nearest generator
    # distance_loss = get_distance_loss(out,data.y,data)
    # print(distance_loss)

    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss

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


def evaluate_batch(data, model, criterion, device='cpu'):
    model.to(device)
    out = model(data)
    loss = criterion(out, data.y)
    return loss