import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#the-messagepassing-base-class
class MessagePassingGNN(MessagePassing):

    class_name = "MessagePassing"

    def __init__(self, input_dim, output_dim, edge_attr_dim, hidden_dim=64, aggr='mean', num_layers=5):
        super(MessagePassingGNN, self).__init__(aggr=aggr)

        # Define your layers here.
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim

        # List to hold the layers.
        self.node_layers = torch.nn.ModuleList()
        self.edge_layers = torch.nn.ModuleList()

        # Input layer
        self.node_layers.append(torch.nn.Linear(input_dim, hidden_dim))
        self.edge_layers.append(torch.nn.Linear(edge_attr_dim[1], hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.node_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.edge_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.node_layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.edge_layers.append(torch.nn.Linear(hidden_dim, output_dim))

        # define a mlp for message passing
        self.mlp_message = nn.Sequential(nn.Linear(hidden_dim * 2, 256), nn.ReLU(), nn.Linear(256, output_dim))
        # define a mlp for update
        self.mlp_update = nn.Sequential(nn.Linear(output_dim, 256), nn.ReLU(), nn.Linear(256, output_dim))

    def forward(self, data):
        # x has shape [num_nodes, input_dim]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_attr_dim]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
        
        # Step 2: Linearly transform node feature matrix.
        for node_layer in self.node_layers[:-1]:
            x = F.relu(node_layer(x))
        
        for edge_attr_layer in self.edge_layers[:-1]:
            edge_attr = F.relu(edge_attr_layer(edge_attr))

        # Step 4-5: Start propagating messages.
        # propage will call the message function, then the aggregate (i.e. mean) function,
        # and finally the update function.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    
    def message(self, x_j, edge_attr):
        # x_j has shape [num_edges, hidden_channels]
        # edge_attr has shape [num_edges, hidden_channels]
        # Combine node features and edge attributes
        combined_features = torch.cat((x_j, edge_attr), dim=-1)

        output = self.mlp_message(combined_features)

        return output

    # aggr_out aggreagtes the messages from the neighbors
    # so we have a message for each node, so we can see the message
    # as a feature for each node, as we include self loops in the graph
    # we are already including the node itself
    def update(self, aggr_out):
        #print("aggr_out", aggr_out.shape)
        output = self.mlp_update(aggr_out)
        # Apply the final layer
        return output
    

    # experiments/ablations:
    # add n of layers in the mlps for message function
    # add biases
    # add dropout
    # add/remove layers to the whole model
    # use attention layer not vanilla mlp
    # add skip/residual connections in update function
    # use only edge features and not node features
