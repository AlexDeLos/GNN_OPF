import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINEConv
from torch_geometric.nn.models import JumpingKnowledge

class GINE(torch.nn.Module):

    class_name = "GINE"
    
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            edge_attr_dim,
            n_hidden_conv=8,
            hidden_conv_dim=16, 
            n_hidden_lin=2,
            hidden_lin_dim=46, 
            dropout_rate=0.2171, 
            jumping_knowledge=True
        ):

        super(GINE, self).__init__()
        self.convs = nn.ModuleList()
        
        for i in range(n_hidden_conv):
            if i == 0:
                hidden_conv_dim = input_dim
            else:
                hidden_conv_dim = hidden_conv_dim
            self.convs.append(
                GINEConv(
                    Sequential(
                        Linear(hidden_conv_dim, hidden_conv_dim), 
                        BatchNorm1d(hidden_conv_dim), 
                        ReLU(),
                        Linear(hidden_conv_dim, hidden_conv_dim), 
                        ReLU()
                    ),
                    edge_dim=edge_attr_dim
                )
            )

        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.JumpingKnowledge.html
        if jumping_knowledge:
            n_of_convs = len(self.convs)
            # if setting mode to 'cat', change the channels of lin1 to hidden_gine_dim * n_of_convs
            self.jumping_knowledge = JumpingKnowledge(mode='lstm', channels=hidden_conv_dim, num_layers=n_of_convs)
        
        self.lins = nn.ModuleList()
        self.lins.append(Linear(hidden_conv_dim, hidden_lin_dim))
        for i in range(n_hidden_lin):
            self.lins.append(Linear(hidden_lin_dim, hidden_lin_dim))
        
        self.lins.append(Linear(hidden_lin_dim, output_dim))

        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        xs = []        
        for conv in self.convs:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            xs.append(x)

        if hasattr(self, 'jumping_knowledge'):
            x = self.jumping_knowledge(xs)
       
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.relu(x, 0.2)
        
        x = self.lins[-1](x)
        
        return x