from torch_geometric.loader import DataLoader as pyg_DataLoader
import torch_geometric.utils as pyg_util
import tqdm
import sys
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
            epoch_val_loss += evaluate_batch(data=batch, model=gnn, criterion=criterion)

        avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
        avg_epoch_val_loss = epoch_val_loss.item() / len(val_dataloader)

        losses.append(avg_epoch_loss)
        val_losses.append(avg_epoch_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.3f}, val_loss: {avg_epoch_val_loss:.3f}')

        # Early stopping
        try:  
            if val_losses[-1]>=val_losses[-2]:
                early_stop += 1
                if early_stop == arguments.patience:
                    print("Early stopping! Epoch:", epoch)
                    break
            else:
                early_stop = 0
        except:
            early_stop = 0
    
    return gnn, losses, val_losses


def train_batch(data, model, optimizer, criterion, physics_crit=True, device='cpu'):
    model.to(device)
    optimizer.zero_grad()
    out = model(data)
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
                        - line length (km)
    @param output:  Model outputs for each node. Node indices expected to match order in input graph.
                    Expected to contain:
                        - Active power p_mw
                        - Reactive power q_mvar
                        - Volt. mag. vm_pu
                        - Volt. angle va_degree
    @param log_loss: Use normal summed absolute imbalances at each node or a logarithmic version.

    @return:    Returns total power imbalance over all the nodes.
    """
    # Calculate admittance values (conductance, susceptance) from impedance values (edges)
    resist_line_total = network.edge_attr[:, 0] * network.edge_attr[:, -1]
    react_line_total = network.edge_attr[:, 1] * network.edge_attr[:, -1]

    denom = resist_line_total * resist_line_total
    denom += react_line_total * react_line_total
    conductances = resist_line_total / denom
    susceptances = -1.0 * react_line_total / denom

    # Combine the fixed input values and predicted missing values
    combined_output = th.zeros(output.shape)

    # slack bus:
    idx_list = (network.x[:, 2] > 0.5)    # get slack node id's
    combined_output[idx_list, 2] += network.x[idx_list, 6]    # Add fixed vm_pu from input; va_degree is 0 for slacks
    combined_output[idx_list, 0] += output[idx_list, 0]    # Add predicted p_mw
    combined_output[idx_list, 1] += output[idx_list, 1]    # Add predicted q_mvar

    # generator + load busses:
    idx_list = (th.logical_and(network.x[:, 0] > 0.5, network.x[:, 1] > 0.5))  # get generator + load node id's
    combined_output[idx_list, 0] += network.x[idx_list, 4]  # Add fixed p_mw from input (already contains value of load p_mw - gen p_mw, so we add instead of subtract)
    combined_output[idx_list, 2] += network.x[idx_list, 6]  # Add fixed vm_pu from input (should be same for both load and gen)
    combined_output[idx_list, 1] += output[idx_list, 1]  # Add predicted q_mvar
    combined_output[idx_list, 3] += output[idx_list, 3]  # Add predicted va_degree

    # generator:
    idx_list = (th.logical_and(network.x[:, 0] < 0.5, network.x[:, 1] > 0.5))  # get generator (not gen + load) node id's
    combined_output[idx_list, 0] -= network.x[idx_list, 4]  # Subtract fixed p_mw from input
                                                            # Because loads and gens both have > 0 power in PandaPower data (gen busses have negative power flow in expected outputs)
    combined_output[idx_list, 2] += network.x[idx_list, 6]  # Add fixed vm_pu from input
    combined_output[idx_list, 1] -= output[idx_list, 1]  # Subtract predicted q_mvar (same reason as above; generators have negative power values in expected outputs)
    combined_output[idx_list, 3] += output[idx_list, 3]  # Add predicted va_degree

    # load + none types (modeled as 0 power demand loads):
    load_no_gen = th.logical_and(network.x[:, 0] > 0.5, network.x[:, 1] < 0.5)
    idx_list = (th.logical_or(th.logical_and(load_no_gen, network.x[:, 2] < 0.5), network.x[:, 3] > 0.5))  # get load + none node id's
    combined_output[idx_list, 0] += network.x[idx_list, 4]  # Add fixed p_mw from input
    combined_output[idx_list, 1] += network.x[idx_list, 5]  # Add fixed q_mvar from input
    combined_output[idx_list, 2] += output[idx_list, 2]  # Add predicted vm_pu
    combined_output[idx_list, 3] += output[idx_list, 3]  # Add predicted va_degree

    # combined_output = output

    # Combine node features with corresponding edges
    from_nodes = pyg_util.select(combined_output, network.edge_index[0], 0)  # list of duplicated node outputs based on edges
    to_nodes = pyg_util.select(combined_output, network.edge_index[1], 0)
    angle_diffs = (from_nodes[:, 3] - to_nodes[:, 3]) * math.pi / 180.0  # list of angle (rad.) differences for all edges

    act_imb = th.abs_(from_nodes.clone()[:, 2]) * th.abs_(to_nodes.clone()[:, 2]) * (conductances * th.cos(angle_diffs) + susceptances * th.sin(angle_diffs))  # per edge power flow into/out of from_nodes
    rea_imb = th.abs_(from_nodes.clone()[:, 2]) * th.abs_(to_nodes.clone()[:, 2]) * (conductances * th.sin(angle_diffs) - susceptances * th.cos(angle_diffs))
    # act_imb = to_nodes[:, 2] * (conductances * th.cos(angle_diffs) + susceptances * th.sin(angle_diffs))  # per edge power flow into/out of from_nodes
    # rea_imb = to_nodes[:, 2] * (conductances * th.sin(angle_diffs) - susceptances * th.cos(angle_diffs))

    aggr_act_imb = pyg_util.scatter(act_imb, network.edge_index[0])  # aggregate all active powers per from_node
    aggr_rea_imb = pyg_util.scatter(rea_imb, network.edge_index[0])

    active_imbalance = combined_output[:, 0] - aggr_act_imb  # subtract from power at each node to find imbalance
    reactive_imbalance = combined_output[:, 1] - aggr_rea_imb
    # active_imbalance = combined_output[:, 0] - combined_output[:, 2] * aggr_act_imb  # subtract from power at each node to find imbalance
    # reactive_imbalance = combined_output[:, 1] - combined_output[:, 2] * aggr_rea_imb

    # Use either sum of absolute imbalances or log of squared imbalances
    if log_loss:
        tot_loss = th.log(1.0 + th.sum(active_imbalance * active_imbalance + reactive_imbalance * reactive_imbalance))
    else:
        tot_loss = th.sum(th.abs(active_imbalance) + th.abs(reactive_imbalance))

    return tot_loss

def evaluate_batch(data, model, criterion, device='cpu'):
    model.to(device)
    out = model(data)
    loss = criterion(out, data.y)
    return loss