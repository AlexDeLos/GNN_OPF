import torch
from utils import train, evaluate, test
from GAT import GAT
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# import data
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

# with open('Data/test.p', 'rb') as handle:
#     tra_dataset_pyg = pickle.load(handle)
# data = tra_dataset_pyg[0]
# instantiate model
model = GAT(dim_input=data.num_features, dim_hidden=[5, 3], dim_output=7, num_layers=2, heads=[1,1])

# setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# train
for epoch in range(1, 150):
    loss = train(data=data, model=model, optimizer=optimizer, criterion=criterion)
    val_loss = evaluate(data=data, model=model, criterion=criterion)
    test_acc = test(data=data, model=model)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, trn_Loss: {loss:.3f}, val_loss: {val_loss:.3f}, tst_acc: {test_acc:.3f}')