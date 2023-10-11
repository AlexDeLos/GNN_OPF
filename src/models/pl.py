import torch as th 
import torch.nn as nn
import torch_geometric.utils as pyg_util
import numpy as np

class ACLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ACLoss, self).__init__()

    def forward(self, inputs, output, edges, attributes):
        # print(inputs)
        # print(output)
        # print(edges.shape, edges)
        # print(attributes.shape, attributes)

        active_imbalance = output[:, 0].clone()
        reactive_imbalance = output[:, 1].clone()

        active_imbalance.requires_grad = True
        reactive_imbalance.requires_grad = True

        from_nodes = pyg_util.select(output, edges[0], 0)
        to_nodes = pyg_util.select(output, edges[1], 0)

        from_nodes.requires_grad = True
        to_nodes.requires_grad = True

        angle_diffs = from_nodes[:, 3] - to_nodes[:, 3] 

        act_imb = th.abs(from_nodes[:, 2]) * th.abs(to_nodes[:, 2]) * (attributes[:, 0] * th.cos(angle_diffs) + attributes[:, 1] * th.sin(angle_diffs))  # per edge power flow into/out of from_nodes
        rea_imb = th.abs(from_nodes[:, 2]) * th.abs(to_nodes[:, 2]) * (attributes[:, 0] * th.sin(angle_diffs) - attributes[:, 1] * th.cos(angle_diffs))

        aggr_act_imb = pyg_util.scatter(act_imb, edges[0])  # aggregate all active powers per from_node
        aggr_rea_imb = pyg_util.scatter(rea_imb, edges[0])

        active_imbalance = output[:, 0] - aggr_act_imb  # subtract from power at each node to find imbalance
        reactive_imbalance = output[:, 1] - aggr_rea_imb

        tot_loss = th.sum(th.abs(active_imbalance) + th.abs(reactive_imbalance))
        # print(tot_loss)
        return tot_loss