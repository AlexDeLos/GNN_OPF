from torch_geometric.loader import DataLoader as pyg_DataLoader
import torch as th
import tqdm
import os
import sys
# local imports
# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_gnn, get_optim, get_criterion, save_model
from utils.utils_hetero import physics_loss_hetero


def train_model_hetero(arguments, train, val):
    output_dims = {node_type: train[0].y_dict[node_type].shape[1] for node_type in train[0].y_dict.keys()}
    edge_types=train[0].edge_index_dict.keys()

    # transform each element in train and val to homogeneous graph
    for i in range(len(train)):
        train[i] = train[i].to_homogeneous()
        
        # 0 is load, 1 is gen, 2 is load_gen, 3 is ext
    for i in range(len(val)):
        val[i] = val[i].to_homogeneous()

    batch_size = arguments.batch_size
    train_dataloader = pyg_DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = pyg_DataLoader(val, batch_size=batch_size, shuffle=False)
    
    gnn_class = get_gnn(arguments.gnn)
    gnn = gnn_class(output_dim_dict=output_dims, edge_types=edge_types)
    epoch_saved = 0
    if arguments.from_checkpoint != None:
        epoch_saved = int(arguments.from_checkpoint.split("_")[-1].split(".")[0])
        gnn.load_state_dict(th.load(arguments.from_checkpoint))
    
    print(f"GNN: \n{gnn}")

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    print(f"Current device: {device}")
    gnn = gnn.to(device)

    optimizer_class = get_optim(arguments.optimizer)
    optimizer = optimizer_class(gnn.parameters(), lr=arguments.learning_rate)
    # scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    criterion = get_criterion(arguments.criterion)

    losses = []
    val_losses = []
    last_batch = None
    min_loss = 1e9
    for epoch in tqdm.tqdm(range(epoch_saved,arguments.n_epochs)): #args epochs range(arguments.n_epochs)
        
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

        if avg_epoch_val_loss < min_loss:
            save_model(gnn, arguments.model_name, best=True)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.6f}, val_loss: {avg_epoch_val_loss:.6f}')
            save_model(gnn, arguments.model_name, epoch=epoch)
        # scheduler.step(avg_epoch_val_loss)

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


#! change the default so that it is easy to change from vector loss to standard loss
def train_batch_hetero(data, model, optimizer, criterion, device='cpu', vector = True, loss_type='standard'):
    data = data.to(device)
    x_dict = {}
    x_load = data.x[data.node_type == 0]
    x_gen = data.x[data.node_type == 1]
    x_load_gen = data.x[data.node_type == 2]
    x_ext = data.x[data.node_type == 3]
    x_dict['load'] = x_load
    x_dict['gen'] = x_gen
    x_dict['load_gen'] = x_load_gen
    x_dict['ext'] = x_ext

    y_dict = {}
    # create mask for each node type
    y_load = data.y[data.node_type == 0]
    y_gen = data.y[data.node_type == 1]
    y_load_gen = data.y[data.node_type == 2]
    y_ext = data.y[data.node_type == 3]
    y_dict['load'] = y_load
    y_dict['gen'] = y_gen
    y_dict['load_gen'] = y_load_gen
    y_dict['ext'] = y_ext


    optimizer.zero_grad()
    out_dict = model(data)

    loss = 0
    if loss_type == 'standard':
        for node_type, y in y_dict.items():
            # prevent nan loss
            if y.shape[0] == 0:
                continue
            if vector:
                loss += vector_loss(out_dict[node_type], y, x=x_dict[node_type], node_type=node_type, device=device)
            else:
                loss += criterion(out_dict[node_type], y)
    else:
        loss = physics_loss_hetero(data, out_dict, device=device)
    loss.backward()
    optimizer.step()
    return loss

#! [0] is angle in degrees [1] is magnitude
#! input angle is in degrees
def vector_loss(out, y, x, node_type, device='cpu'):
    if node_type == 'load':
        vec_mag_and_vec_angle = out
    if node_type == 'gen':
        # stack the output y and the second column of input x
        vec_mag_and_vec_angle = th.stack((out[:,0], x[:,1]), dim=1)
    if node_type == 'load_gen':
        # stack the output y and the third column of input x
        vec_mag_and_vec_angle = th.stack((out[:,0], x[:,1]), dim=1)
    
    out_x = th.mul(th.cos(th.deg2rad(vec_mag_and_vec_angle[:,0])), vec_mag_and_vec_angle[:,1])
    out_y = th.mul(th.sin(th.deg2rad(vec_mag_and_vec_angle[:,0])), vec_mag_and_vec_angle[:,1])
    out_vector = th.stack((out_x, out_y))

    if node_type == 'load':
        data_mag_and_data_angle = y
    if node_type == 'gen':
        data_mag_and_data_angle = th.stack((y[:,0], x[:,1]), dim=1)
    if node_type == 'load_gen':
        data_mag_and_data_angle = th.stack((y[:,0], x[:,1]), dim=1)
        
    data_x = th.mul(data_mag_and_data_angle[:,1], th.cos(th.deg2rad(data_mag_and_data_angle[:,0])))
    data_y = th.mul(data_mag_and_data_angle[:,1], th.sin(th.deg2rad(data_mag_and_data_angle[:,0])))
    data_vector = th.stack((data_x, data_y))


    loss = th.mean(distance(out_vector, data_vector))
    return loss

def distance(a,b):
    pdist = th.nn.PairwiseDistance(p=2)
    return pdist(a.T,b.T)


#? Combine both vec loss and loss
#? add more weight to the magnitude?
#? what are the angle units???
def evaluate_batch_hetero(data, model, criterion, device='cpu', vector = True, loss_type='standard'):
    data = data.to(device)
    x_dict = {}
    # create mask for each node type
    x_load = data.x[data.node_type == 0]
    x_gen = data.x[data.node_type == 1]
    x_load_gen = data.x[data.node_type == 2]
    x_ext = data.x[data.node_type == 3]
    x_dict['load'] = x_load
    x_dict['gen'] = x_gen
    x_dict['load_gen'] = x_load_gen
    x_dict['ext'] = x_ext

    y_dict = {}
    # create mask for each node type
    y_load = data.y[data.node_type == 0]
    y_gen = data.y[data.node_type == 1]
    y_load_gen = data.y[data.node_type == 2]
    y_ext = data.y[data.node_type == 3]
    y_dict['load'] = y_load
    y_dict['gen'] = y_gen
    y_dict['load_gen'] = y_load_gen
    y_dict['ext'] = y_ext

    out_dict = model(data)
    loss = 0
    if loss_type == 'standard':
        for node_type, y in y_dict.items():
            # prevent nan loss
            if y.shape[0] == 0:
                continue
            if vector:
                y = th.nan_to_num(y)
                ret= vector_loss(out_dict[node_type], y, x_dict[node_type], node_type=node_type, device=device)
                if th.isnan(ret):
                    print("nan loss")
                else:
                    loss += ret
            else:
                loss += criterion(out_dict[node_type], y)
    else:
        loss = physics_loss_hetero(data, out_dict, device=device)
    return loss