from torch_geometric.loader import DataLoader as pyg_DataLoader
import tqdm
from utils import get_gnn, get_optim, get_criterion
from models.pl import ACLoss


def train_model(arguments, train, val):
    input_dim = train[0].x.shape[1]
    edge_attr_dim = train[0].edge_attr.shape[1] # why not [1]
    output_dim = train[0].y.shape[1]

    print(f"Input shape: {input_dim}\nOutput shape: {output_dim}")

    batch_size = arguments.batch_size
    train_dataloader = pyg_DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = pyg_DataLoader(val, batch_size=batch_size, shuffle=False)
    gnn_class = get_gnn(arguments.gnn)
    gnn = gnn_class(input_dim, output_dim, edge_attr_dim)
    print(f"GNN: \n{gnn}")
    ac = ACLoss()

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
            epoch_loss += train_batch(data=batch, model=gnn, optimizer=optimizer, criterion=criterion, ac=ac)
        gnn.eval()
        for batch in val_dataloader:
            epoch_val_loss += evaluate_batch(data=batch, model=gnn, criterion=criterion, ac=ac)

        avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
        avg_epoch_val_loss = epoch_val_loss.item() / len(val_dataloader)

        losses.append(avg_epoch_loss)
        val_losses.append(avg_epoch_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.6f}, val_loss: {avg_epoch_val_loss:.6f}')
        
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
    
    return gnn, losses, val_losses


def train_batch(data, model, optimizer, criterion, device='cpu', ac=None):
    model.to(device)
    optimizer.zero_grad()
    out = model(data)
    loss = ac(out, data.y, data.edge_index, data.edge_attr) # criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss


def evaluate_batch(data, model, criterion, device='cpu', ac=None):
    model.to(device)
    out = model(data)
    loss = ac(out, data.y, data.edge_index, data.edge_attr) # criterion(out, data.y)
    return loss