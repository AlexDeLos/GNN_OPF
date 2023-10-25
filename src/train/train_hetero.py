from torch_geometric.loader import DataLoader as pyg_DataLoader
import torch as th
import tqdm
import os
import sys
# local imports
# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_gnn, get_optim, get_criterion
from utils.utils_hetero import physics_loss_hetero


def train_model_hetero(arguments, train, val):
    output_dims = {node_type: train[0].y_dict[node_type].shape[1] for node_type in train[0].y_dict.keys()}

    batch_size = arguments.batch_size
    train_dataloader = pyg_DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = pyg_DataLoader(val, batch_size=batch_size, shuffle=False)
    
    gnn_class = get_gnn(arguments.gnn)

    gnn = gnn_class(output_dim_dict=output_dims, edge_types=train[0].edge_index_dict.keys())
    
    print(f"GNN: \n{gnn}")

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    print(f"Current device: {device}")
    gnn = gnn.to(device)

    optimizer_class = get_optim(arguments.optimizer)
    optimizer = optimizer_class(gnn.parameters(), lr=arguments.learning_rate)
    criterion = get_criterion(arguments.criterion)

    losses = []
    val_losses = []
    last_batch = None
    for epoch in tqdm.tqdm(range(arguments.n_epochs)): #args epochs range(arguments.n_epochs)
        epoch_loss = 0.0
        epoch_val_loss = 0.0
        gnn.train()
        for batch in train_dataloader:
            epoch_loss += train_batch_hetero(data=batch, model=gnn, optimizer=optimizer, criterion=criterion, loss_type=arguments.loss_type, device=device)
        gnn.eval()
        for batch in val_dataloader:
            epoch_val_loss += evaluate_batch_hetero(data=batch, model=gnn, criterion=criterion, loss_type=arguments.loss_type, device=device)

        avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
        avg_epoch_val_loss = epoch_val_loss.item() / len(val_dataloader)

        losses.append(avg_epoch_loss)
        val_losses.append(avg_epoch_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.6f}, val_loss: {avg_epoch_val_loss:.6f}')

        # Early stopping
        try:  
            if val_losses[-1] >= val_losses[-2]:
                early_stop += 1
                if early_stop == arguments.patience:
                    print("Early stopping! Epoch:", epoch)
                    break
            else:
                early_stop = 0
        except:
            early_stop = 0
    print(f"Min loss: {min(val_losses)} last loss: {val_losses[-1]}")
    return gnn, losses, val_losses, last_batch


def train_batch_hetero(data, model, optimizer, criterion, device='cpu', loss_type='standard'):
    data = data.to(device)
    optimizer.zero_grad()
    out_dict = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    loss = 0
    if loss_type == 'standard':
        for node_type, y in data.y_dict.items():
            # prevent nan loss
            if y.shape[0] == 0:
                continue
            loss += criterion(out_dict[node_type], y)
    else:
        loss = physics_loss_hetero(data, out_dict, device=device)
    loss.backward()
    optimizer.step()
    return loss


def evaluate_batch_hetero(data, model, criterion, device='cpu', loss_type='standard'):
    data = data.to(device)
    out_dict = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    loss = 0
    if loss_type == 'standard':
        for node_type, y in data.y_dict.items():
            # prevent nan loss
            if y.shape[0] == 0:
                continue
            loss += criterion(out_dict[node_type], y)
    else:
        loss = physics_loss_hetero(data, out_dict, device=device)
    return loss