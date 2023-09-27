import torch
from utils import train_single_graph, evaluate_single_graph, test_single_graph, train_epoch, evaluate_epoch, test_epoch
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import PPI
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import DataLoader

"""# single graph training
# dataset contains 1 graph
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

# with open('Data/test.p', 'rb') as handle:
#     tra_dataset_pyg = pickle.load(handle)
# data = tra_dataset_pyg[0]
# instantiate model
model = GAT(in_dim=data.num_features, hidden_dim=[5, 3], out_dim=7, num_layers=2, heads=[1,1])
#model = GraphSAGE(in_dim=data.num_features, hidden_dim=5, out_dim=7, num_layers=2, dropout=0.2)
# setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# train
for epoch in range(1, 150):
    loss = train_single_graph(data=data, model=model, optimizer=optimizer, criterion=criterion)
    val_loss = evaluate_single_graph(data=data, model=model, criterion=criterion)
    test_acc = test_single_graph(data=data, model=model)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, trn_Loss: {loss:.3f}, val_loss: {val_loss:.3f}, tst_acc: {test_acc:.3f}')"""

#############################################################################################################

# multi graph training

# use ppi dataset
"""train_dataset = ...
val_dataset = ...
test_dataset = ...


# instantiate model
model = GAT(dim_input=train_dataset.num_features, dim_hidden=[5, 3], dim_output=train_dataset.num_classes, num_layers=2, heads=[1,1])

# setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# setup dataloaders for trian, val and test
train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# train
for epoch in range(1, 5):
    loss = train_epoch(model=model, loader=train_data_loader, optimizer=optimizer, criterion=criterion)
    val_loss = evaluate_epoch(model=model, loader=val_data_loader, criterion=criterion)
    test_acc = test_epoch(model=model, loader=test_data_loader)
    
    print(f'Epoch: {epoch:03d}, trn_Loss: {loss:.3f}, val_loss: {val_loss:.3f}, tst_acc: {test_acc:.3f}')"""




