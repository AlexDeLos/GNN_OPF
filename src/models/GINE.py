import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool

class GINE(torch.nn.Module):

    class_name = "GINE"
    
    def __init__(self, input_dim, output_dim, edge_dim, hidden_gine_dim=16, hidden_lin_dim=64):
        super(GINE, self).__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(input_dim, hidden_gine_dim),
                       BatchNorm1d(hidden_gine_dim), ReLU(),
                       Linear(hidden_gine_dim, hidden_gine_dim), ReLU()),
                       edge_dim=edge_dim) #[1]
        self.conv2 = GINEConv(
            Sequential(Linear(hidden_gine_dim, hidden_gine_dim), BatchNorm1d(hidden_gine_dim), ReLU(),
                       Linear(hidden_gine_dim, hidden_gine_dim), ReLU()),
                       edge_dim=edge_dim) #[1]
        self.conv3 = GINEConv(
            Sequential(Linear(hidden_gine_dim, hidden_gine_dim), BatchNorm1d(hidden_gine_dim), ReLU(),
                       Linear(hidden_gine_dim, hidden_gine_dim), ReLU()),
                       edge_dim=edge_dim) #[1]
        self.lin1 = Linear(hidden_gine_dim*3, hidden_lin_dim)
        self.lin2 = Linear(hidden_lin_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = data.edge_attr

        # Node embeddings 
        h1 = self.conv1(x, edge_index=edge_index, edge_attr=edge_attr)
        h2 = self.conv2(h1, edge_index=edge_index, edge_attr=edge_attr)
        h3 = self.conv3(h2, edge_index=edge_index, edge_attr=edge_attr)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h