import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# Define your custom GAT layer
class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.5):
        super(GATLayer, self).__init__(aggr='add')  # "add" aggregation method
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, heads * out_channels)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.dropout = nn.Dropout(p=dropout)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight.data)
        nn.init.xavier_uniform_(self.att.data)

    def forward(self, x, edge_index):
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients
        alpha = torch.cat([x_i, x_j], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # Normalize attention scores using softmax
        alpha = alpha / degree(edge_index_i, size=size_i, dtype=x_i.dtype).view(-1, 1)

        return {'x_j': x_j, 'alpha': alpha}

    def update(self, aggr_out):
        # Combine neighbor node features weighted by attention coefficients
        return aggr_out

# Define your GAT-based regression model
class GATNodeRegression(nn.Module):
    def __init__(self, in_channels, output_size, out_channels=16, num_heads=4, num_layers=4, dropout=0.5):
        super(GATNodeRegression, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.num_layers = num_layers

        for _ in range(num_layers):
            self.conv_layers.append(GATLayer(in_channels, out_channels, num_heads, dropout))
            in_channels = num_heads * out_channels

        self.fc = nn.Linear(in_channels, output_size)  # Output is a scalar for regression

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer in self.conv_layers:
            x = layer(x, edge_index)

        x = self.fc(x)

        return x
