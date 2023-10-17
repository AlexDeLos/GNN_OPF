import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.models import JumpingKnowledge

class GINE(torch.nn.Module):

    class_name = "GINE"
    

    def __init__(self, input_dim, output_dim, edge_dim, hidden_gine_dim=16, hidden_lin_dim=64, dropout_rate=0.5, jumping_knowledge=False, n_of_convs_in=1):
        super(GINE, self).__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(input_dim, hidden_gine_dim),
                       BatchNorm1d(hidden_gine_dim), ReLU(),
                       Linear(hidden_gine_dim, hidden_gine_dim), ReLU()),
                       edge_dim=edge_dim) #[1]
        
        self.convs2 = nn.ModuleList()
        for _ in range(n_of_convs_in):
            self.convs2.append(GINEConv(
                Sequential(Linear(hidden_gine_dim, hidden_gine_dim), BatchNorm1d(hidden_gine_dim), ReLU(),
                        Linear(hidden_gine_dim, hidden_gine_dim), ReLU()),
                        edge_dim=edge_dim)) #[1]
        self.conv3 = GINEConv(
            Sequential(Linear(hidden_gine_dim, hidden_gine_dim), BatchNorm1d(hidden_gine_dim), ReLU(),
                       Linear(hidden_gine_dim, hidden_gine_dim), ReLU()),
                       edge_dim=edge_dim) #[1]

        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.JumpingKnowledge.html
        if jumping_knowledge:
            n_of_convs = n_of_convs_in + 2
            # if setting mode to 'cat', change the channels of lin1 to hidden_gine_dim * n_of_convs
            self.jumping_knowledge = JumpingKnowledge(mode='lstm', channels=hidden_gine_dim, num_layers=n_of_convs)
        self.lin1 = Linear(hidden_gine_dim, hidden_lin_dim)
        self.lin2 = Linear(hidden_lin_dim, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = data.edge_attr
        xs = []
        # Node embeddings 
        h1 = self.conv1(x, edge_index=edge_index, edge_attr=edge_attr)
        xs.append(h1)
        x = h1
        for c in self.convs2:
            x = c(x, edge_index=edge_index, edge_attr=edge_attr)
            xs.append(x)
        # h2 = self.convs2(h1, edge_index=edge_index, edge_attr=edge_attr)
        xs.append(x)
        h3 = self.conv3(x, edge_index=edge_index, edge_attr=edge_attr)
        xs.append(h3)

        if hasattr(self, 'jumping_knowledge'):
            h = self.jumping_knowledge(xs)
        else:
            h = h3

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.lin2(h)
        
        return h