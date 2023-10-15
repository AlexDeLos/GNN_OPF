from torch_geometric.loader import DataLoader as pyg_DataLoader
import tqdm
import	torch as th
from utils import get_gnn, get_optim, get_criterion
from models.pl import ACLoss    
import optuna

def train_model(arguments, train, val):
    input_dim = train[0].x.shape[1]
    edge_attr_dim = train[0].edge_attr.shape[1]
    output_dim = train[0].y.shape[1]

    print(f"Input shape: {input_dim}\nOutput shape: {output_dim}")

    batch_size = arguments.batch_size
    train_dataloader = pyg_DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = pyg_DataLoader(val, batch_size=batch_size, shuffle=False)
    gnn_class = get_gnn(arguments.gnn)
    gnn = gnn_class(input_dim,
                    output_dim, 
                    edge_attr_dim, 
                    arguments.n_hidden_gnn, 
                    arguments.gnn_hidden_dim, 
                    arguments.n_hidden_lin, 
                    arguments.lin_hidden_dim)
    print(f"GNN: \n{gnn}")

    optimizer_class = get_optim(arguments.optimizer)
    optimizer = optimizer_class(gnn.parameters(), lr=arguments.learning_rate)
    criterion = get_criterion(arguments.criterion)

    losses = []
    val_losses = []
    last_batch = None
    for epoch in range(arguments.n_epochs): #tqdm.tqdm(range(arguments.n_epochs)): #args epochs
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

        # if epoch % 10 == 0:
        #     print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.6f}, val_loss: {avg_epoch_val_loss:.6f}')
        
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


def train_batch(data, model, optimizer, criterion, device='cpu'):
    model.to(device)
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y) # ac(out, data.x, data.edge_index, data.edge_attr)
    loss.backward()
    optimizer.step()
    return loss


def evaluate_batch(data, model, criterion, device='cpu'):
    model.to(device)
    out = model(data)
    loss = criterion(out, data.y) # ac(out, data.x, data.edge_index, data.edge_attr)
    return loss


def train_model_hetero(arguments, train, val):
    # edge_attr_dims = {edge_type: train[0].edge_index_dict[edge_type].shape[1] for edge_type in train[0].edge_index_dict.keys()}
    output_dims = {node_type: train[0].y_dict[node_type].shape[1] for node_type in train[0].y_dict.keys()}

    batch_size = arguments.batch_size
    train_dataloader = pyg_DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = pyg_DataLoader(val, batch_size=batch_size, shuffle=False)
    
    gnn_class = get_gnn(arguments.gnn)

    gnn = gnn_class(hidden_channels=arguments.gnn_hidden_dim,
                    num_layers=arguments.n_hidden_gnn,
                    edge_types=train[0].edge_index_dict.keys(),
                    out_channels_dict=output_dims)
    
    print(f"GNN: \n{gnn}")

    optimizer_class = get_optim(arguments.optimizer)
    optimizer = optimizer_class(gnn.parameters(), lr=arguments.learning_rate)
    criterion = get_criterion(arguments.criterion)

    losses = []
    val_losses = []
    last_batch = None
    for epoch in range(arguments.n_epochs):
        epoch_loss = 0.0
        epoch_val_loss = 0.0
        gnn.train()
        for batch in train_dataloader:
            epoch_loss += train_batch_hetero(data=batch, model=gnn, optimizer=optimizer, criterion=criterion)
        gnn.eval()
        for batch in val_dataloader:
            epoch_val_loss += evaluate_batch_hetero(data=batch, model=gnn, criterion=criterion)

        avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
        avg_epoch_val_loss = epoch_val_loss.item() / len(val_dataloader)

        losses.append(avg_epoch_loss)
        val_losses.append(avg_epoch_val_loss)

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


def train_batch_hetero(data, model, optimizer, criterion, device='cpu'):
    model.to(device)
    optimizer.zero_grad()
    out_dict = model(data.x_dict, data.edge_index_dict)
    loss = 0
    for node_type, y in data.y_dict.items():
        loss += criterion(out_dict[node_type], y)
    loss.backward()
    optimizer.step()
    return loss


def evaluate_batch_hetero(data, model, criterion, device='cpu'):
    model.to(device)
    out_dict = model(data.x_dict, data.edge_index_dict)
    loss = 0
    for node_type, y in data.y_dict.items():
        loss += criterion(out_dict[node_type], y)
    return loss