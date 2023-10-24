import torch as th
from torch_geometric.loader import DataLoader as pyg_DataLoader
import tqdm
import os
import sys
import numpy as np
# local imports
# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_gnn, get_optim, get_criterion
from utils.utils_homo import pretrain
from utils.utils_physics import physics_loss


def train_model(arguments, train, val):
    input_dim = train[0].x.shape[1]
    edge_attr_dim = train[0].edge_attr.shape[1]
    output_dim = train[0].y.shape[1]

    print(f"Input shape: {input_dim}\nOutput shape: {output_dim}")

    batch_size = int(arguments.batch_size)
    train_dataloader = pyg_DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = pyg_DataLoader(val, batch_size=batch_size, shuffle=False)
    gnn_class = get_gnn(arguments.gnn)

    if arguments.pretrain:
        print("Pretraining with Deep Graph Infomax...")
        gnn = pretrain(gnn_class, input_dim, output_dim, edge_attr_dim, train_dataloader)
        print("Pretraining done")
    else:
        gnn = gnn_class(
            input_dim,
            output_dim, 
            edge_attr_dim
        )
        
    print(f"GNN: \n{gnn}")
    ac = None

    optimizer_class = get_optim(arguments.optimizer)
    optimizer = optimizer_class(gnn.parameters(), lr=float(arguments.learning_rate), weight_decay=arguments.weight_decay)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    criterion = get_criterion(arguments.criterion)

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    print(f"Current device: {device}")
    gnn = gnn.to(device)

    losses = []
    val_losses = []
    last_batch = None
    for epoch in tqdm.tqdm(range(int(arguments.n_epochs))):
        epoch_loss = 0.0
        epoch_val_loss = 0.0
        gnn.train()
        for batch in train_dataloader:
            epoch_loss += train_batch(data=batch, model=gnn, optimizer=optimizer, criterion=criterion, loss_type=arguments.loss_type, mix_weight=arguments.mixed_loss_weight, device=device)
        gnn.eval()
        for batch in val_dataloader:
            epoch_val_loss += evaluate_batch(data=batch, model=gnn, criterion=criterion, loss_type=arguments.loss_type, device=device)

        avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
        avg_epoch_val_loss = epoch_val_loss.item() / len(val_dataloader)

        losses.append(avg_epoch_loss)
        val_losses.append(avg_epoch_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.6f}, val_loss: {avg_epoch_val_loss:.6f}')
        scheduler.step(avg_epoch_val_loss)
        #Early stopping
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
    print(f"Min loss: {min(val_losses)} last loss: {val_losses[-1]}")
    return gnn, losses, val_losses, last_batch


def train_batch(data, model, optimizer, criterion, loss_type='standard', mix_weight=0.1, device='cpu'):
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data)
    if loss_type != 'standard':
        loss1 = physics_loss(data, out, log_loss=True, device=device)

        if loss_type == 'mixed':
            loss2 = criterion(out, data.y)
            loss = loss1 + mix_weight * loss2
        else:
            loss = loss1
    elif vector_loss:
        loss = vector_loss(out, data, criterion)
    else:
        loss = criterion(out, data)
        
    loss.backward()
    optimizer.step()
    return loss

def vector_loss(data,out, device='cpu'):
    vec_mag_and_vec_angle = out.y[:,:2]
    out_x = th.mul(th.cos(vec_mag_and_vec_angle[:,0]), vec_mag_and_vec_angle[:,1])
    out_y = th.mul(th.sin(vec_mag_and_vec_angle[:,0]), vec_mag_and_vec_angle[:,1])
    out_vector = th.stack((out_x, out_y))

    data_mag_and_data_angle = data[:,:2]
    data_x = th.cos(data_mag_and_data_angle[:,0]) * data_mag_and_data_angle[:,1]
    data_y = th.sin(data_mag_and_data_angle[:,0]) * data_mag_and_data_angle[:,1]
    data_vector = th.stack((data_x, data_y))


    loss = th.mean(distance(out_vector, data_vector))
    return loss

def distance(a,b):
    return th.sum(th.subtract(a,b)**2,dim=1)

def evaluate_batch(data, model, criterion, device='cpu', loss_type='standard'):
    data = data.to(device)
    out = model(data)
    if loss_type != 'standard':
        loss = physics_loss(data, out, log_loss=False, device=device)
    else:
        loss = criterion(out, data.y) # ac(out, data.x, data.edge_index, data.edge_attr)
    return loss
