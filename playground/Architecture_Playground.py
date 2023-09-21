# imports:
import time
import torch
import pickle
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.nn import TAGConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader as pyg_DataLoader
# END Imports


#Importing Data
with open('Data/test.p', 'rb') as handle:
    tra_dataset_pyg = pickle.load(handle)
with open('Data/toy_validation_dataset.p', 'rb') as handle:
    val_dataset_pyg = pickle.load(handle)
with open('Data/toy_test_dataset.p', 'rb') as handle:
    tst_dataset_pyg = pickle.load(handle)

print('Number of training examples:',   len(tra_dataset_pyg))
print('Number of validation examples:', len(val_dataset_pyg))
print('Number of test examples:',       len(tst_dataset_pyg))

# Check one example to see what the data looks like
print(tra_dataset_pyg[0])
#tra_dataset_pyg[0]
# OutPut: Data(x=[37, 4], edge_attr=[116, 1], edge_index=[2, 116], y=[37, 1])
# Our "OLD" Output: Data(x=[5, 3], edge_index=[2, 12], edge_attr=[12, 2], y=[5, 1], weight=[12], path=[12])
# Our NEW output: Data(x=[5, 3], edge_index=[2, 12], edge_attr=[12, 2], y=[5, 1])

# Define the GNNs
class GNN_Example(nn.Module):
  """
    This class defines a PyTorch module that takes in a graph represented in the PyTorch Geometric Data format,
    and outputs a tensor of predictions for each node in the graph. The model consists of one or more TAGConv layers,
    which are a type of graph convolutional layer.

    Args:
        node_dim (int): The number of node inputs.
        edge_dim (int): The number of edge inputs.
        output_dim (int, optional): The number of outputs (default: 1).
        hidden_dim (int, optional): The number of hidden units in each GNN layer (default: 50).
        n_gnn_layers (int, optional): The number of GNN layers in the model (default: 1).
        K (int, optional): The number of hops in the neighbourhood for each GNN layer (default: 2).
        dropout_rate (float, optional): The dropout rate to be applied to the output of each GNN layer (default: 0).

    """
  def __init__(self, node_dim, edge_dim, output_dim=1, hidden_dim=50, n_gnn_layers=1, K=2, dropout_rate=0):
    super().__init__()
    self.node_dim = node_dim          
    self.edge_dim = edge_dim          
    self.output_dim = output_dim      
    self.hidden_dim = hidden_dim      
    self.n_gnn_layers = n_gnn_layers  
    self.K = K                        
    self.dropout_rate = dropout_rate
    
    self.convs = nn.ModuleList()

    if n_gnn_layers == 1:
      self.convs.append(TAGConv(node_dim, output_dim, K=K))
    else:
      self.convs.append(TAGConv(node_dim, hidden_dim, K=K))

      for l in range(n_gnn_layers-2):
          self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
          
      self.convs.append(TAGConv(hidden_dim, output_dim, K=K))

  def forward(self, data):
      """Applies the GNN to the input graph.

        Args:
            data (Data): A PyTorch Geometric Data object representing the input graph.

        Returns:
            torch.Tensor: The output tensor of the GNN.

        """
      x = data.x
      edge_index = data.edge_index
      edge_attr = data.edge_attr
      print(x.shape)
      print(self.convs[0])
      for i in range(len(self.convs)-1):
          #TODO: Fix the dimensions so that we do not get an error?
          x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
          x = nn.Dropout(self.dropout_rate, inplace=False)(x)
          x = nn.PReLU()(x)
      
      x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
      # x = nn.Sigmoid()(x)
      
      return x
  
# END of GNNs decleration section


#START of model initialization

# Set model parameters
node_dim =   tra_dataset_pyg[0].x.shape[1]
edge_dim =   tra_dataset_pyg[0].edge_attr.shape[1]
output_dim = tra_dataset_pyg[0].y.shape[1]
hidden_dim = 16
n_gnn_layers = 3
K=1
dropout_rate = 0

# Create model
model = GNN_Example(node_dim, edge_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate)
print(model)


# Set model parameters
node_dim =   tra_dataset_pyg[0].x.shape[1]
edge_dim =   tra_dataset_pyg[0].edge_attr.shape[1]
output_dim = tra_dataset_pyg[0].y.shape[1]
hidden_dim = 16
n_gnn_layers = 3
K=1
dropout_rate = 0.1

# Create model
model = GNN_Example(node_dim, edge_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate)
print(model)



#START of Training section

def train_epoch(model, loader, optimizer, device='cpu'):
  """
    Trains a neural network model for one epoch using the specified data loader and optimizer.

    Args:
        model (nn.Module): The neural network model to be trained.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the training data.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer used for training the model.
        device (str): The device used for training the model (default: 'cpu').

    Returns:
        float: The mean loss value over all the batches in the DataLoader.

  """
  model.train()
  model.to(device)

  total_loss = 0.0

  for batch in loader:
    batch = batch.to(device)
    optimizer.zero_grad()
    output = model(batch)
    loss = nn.MSELoss()(output, batch.y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  
  return total_loss / len(loader)


def evaluate_epoch(model, loader, device='cpu'):
    """
    Evaluates the performance of a trained neural network model on a dataset using the specified data loader.

    Args:
        model (nn.Module): The trained neural network model to be evaluated.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the evaluation data.
        device (str): The device used for evaluating the model (default: 'cpu').

    Returns:
        float: The mean loss value over all the batches in the DataLoader.

    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        loss = nn.MSELoss()(output, batch.y)
        total_loss += loss.item()

    return total_loss / len(loader)

# Optimize the model

#Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set training parameters
learning_rate = 0.001
batch_size = 16
num_epochs = 120

# Create the optimizer to train the neural network via back-propagation
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# Create the training and validation dataloaders to "feed" data to the GNN in batches
tra_loader = pyg_DataLoader(tra_dataset_pyg, batch_size=batch_size, shuffle=True)
val_loader = pyg_DataLoader(val_dataset_pyg, batch_size=batch_size, shuffle=False)


# START OF TRAIN LOOP PROPER:
#create vectors for the training and validation loss
train_losses = []
val_losses = []
patience = 5       # patience for early stopping

#start measuring time
start_time = time.time()

for epoch in range(1, num_epochs+1):
    # Model training
    train_loss = train_epoch(model, tra_loader, optimizer, device=device)

    # Model validation
    val_loss = evaluate_epoch(model, val_loader, device=device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Early stopping
    try:
        if val_losses[-1]>=val_losses[-2]:
            early_stop += 1
            if early_stop == patience:
                print("Early stopping! Epoch:", epoch)
                break
        else:
            early_stop = 0
    except:
        early_stop = 0

    if epoch%10 == 0:
        print("epoch:",epoch, "\t training loss:", np.round(train_loss,4),
                            "\t validation loss:", np.round(val_loss,4))

elapsed_time = time.time() - start_time
print(f'Model training took {elapsed_time:.3f} seconds')


#DISPLAY RESULTS SECTION:

# plot the training and validation loss curves
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()