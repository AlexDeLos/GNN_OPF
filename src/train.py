from torch_geometric.loader import DataLoader as pyg_DataLoader
import tqdm
import torch as th
import numpy as np
from utils import get_gnn, get_optim, get_criterion


def train_model(arguments, train, val, test):
    input_dim = train[0].x.shape[1]
    edge_attr_dim = train[0].edge_attr.shape # why not [1]
    output_dim = train[0].y.shape[1]

    print(f"Input shape: {input_dim}\nOutput shape: {output_dim}")

    batch_size = arguments.batch_size
    train_dataloader = pyg_DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = pyg_DataLoader(val, batch_size=batch_size, shuffle=True)
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
            epoch_val_loss += evaluate_batch(data=batch, model=gnn, criterion=criterion)

        avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
        avg_epoch_val_loss = epoch_val_loss.item() / len(val_dataloader)

        losses.append(avg_epoch_loss)
        val_losses.append(avg_epoch_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.3f}, val_loss: {avg_epoch_val_loss:.3f}')
    
    return gnn, losses, val_losses


def train_batch(data, model, optimizer, criterion, physics_crit=True, device='cpu'):
    model.to(device)
    optimizer.zero_grad()
    out = model(data)
    # out_clone = out.clone().detach()
    if physics_crit:
        loss = physics_loss(data, out)
    else:
        loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss


def physics_loss(network, output, log_loss=True):
    """
    Calculates power imbalances at each node in the graph and sums results.
    Based on loss from https://arxiv.org/abs/2204.07000

    @param network:    Input graph used for the NN model.
                    Expected to contain nodes list and edges between nodes with features:
                        - resistance r over the line
                        - reactance x over the line
    @param output:  Model outputs for each node. Node indices expected to match order in input graph.
                    Expected to contain:
                        - Active power p_mw
                        - Reactive power q_mvar
                        - Volt. mag. vm_pu
                        - Volt. angle va_degree
    @param log_loss: Use normal summed absolute imbalances at each node or a logarithmic version.

    @return:    Returns total power imbalance over all the nodes.
    """
    # Get predicted power levels from the model outputs
    # active_imbalance = output.p_mw #output[:, 0]
    # reactive_imbalance = output.q_mvar #output[:, 1]
    active_imbalance = output[:,0] # th.zeros(output_r.shape[0])
    reactive_imbalance = output[:,1]#th.zeros(output_r.shape[0])
    #output = [[0,1,2,3]]*network.num_edges
    #output[:][2] = output_r

    # Calculate admittance values (conductance, susceptance) from impedance values (edges)
    # edge_att[:, 0] should contain resistances r, edge_att[:, 1] should contain reactances x, edge_attr[:,-1] line length (km)
    resist_line_total = network.edge_attr[:, 0] * network.edge_attr[:, -1]
    react_line_total = network.edge_attr[:, 1] * network.edge_attr[:, -1]

    denom = resist_line_total * resist_line_total
    denom += react_line_total * react_line_total
    conductances = resist_line_total / denom
    susceptances = -1.0 * react_line_total / denom

    # Go over all edges and update the power imbalances for each node accordingly
    # TODO: way to do this with tensors instead of loop?
    for i, x in enumerate(th.transpose(network.edge_index, 0, 1)):
        # x contains node indices [from, to]
        angle_diff = output[x[0],3] - output[x[1],3]

        # active_imbalance[x[0]] -= th.abs(output[x[0],2]).detach().numpy() * th.abs(output[x[1],2]).detach().numpy() \
        #                             * (conductances[i] * th.cos(angle_diff).detach().numpy() + susceptances[i] * th.sin(angle_diff)).detach().numpy()
        # reactive_imbalance[x[0]] -= th.abs(output[x[0],2]).detach().numpy() * th.abs(output[x[1],2]).detach().numpy() \
        #                             * (conductances[i] * th.sin(angle_diff).detach().numpy() - susceptances[i] * th.cos(angle_diff)).detach().numpy()

        # TODO: fix detach issue leading to inplace modifications
        active_imbalance[x[0]] -= th.abs_(output[x[0], 2]).detach() * th.abs_(output[x[1], 2]).detach() \
                                  * (conductances[i] * th.cos(angle_diff) + susceptances[i] * th.sin(angle_diff))
        reactive_imbalance[x[0]] -= th.abs_(output[x[0], 2]).detach() * th.abs_(output[x[1], 2]).detach() \
                                    * (conductances[i] * th.sin(angle_diff) - susceptances[i] * th.cos(angle_diff))

    # Use either sum of absolute imbalances or log of squared imbalances
    if log_loss:
        tot_loss = th.log(1.0 + th.sum(active_imbalance * active_imbalance + reactive_imbalance * reactive_imbalance))
    else:
        tot_loss = th.sum(np.abs(active_imbalance) + np.abs(reactive_imbalance))

    return tot_loss

def evaluate_batch(data, model, criterion, device='cpu'):
    model.to(device)
    out = model(data)
    loss = criterion(out, data.y)
    return loss