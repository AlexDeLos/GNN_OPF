import pickle
import pandapower.networks as pn
import pandapower.plotting as ppl
import pandapower as pp
import numpy as np

import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
# We pick the case5 network as an example TODO: Change to the actual network that we use
net = pn.case5()

# From the pandapower network, to a networkx graph
G = ppl.create_nxgraph(net, respect_switches = False)

# Iterate over the nodes (buses), and add the attributes to the networkx graph as x 
for node in net.bus.itertuples():

    # https://pandapower.readthedocs.io/en/v2.1.0/elements/bus.html#input-paramaters
    G.nodes[node.Index]['x'] = [float(node.vn_kv),float(node.max_vm_pu),float(node.min_vm_pu)]

    # Label, what we are going to predict?
    G.nodes[node.Index]['y'] = [float(0)] #TODO: What on earth is the label?

# Add edge attributes (attribute is the impedance of the line)
for edge in net.line.itertuples():
    G.edges[edge.from_bus, edge.to_bus, ('line', edge.Index)]['edge_attr'] = [float(edge.r_ohm_per_km * edge.length_km)]

#turn the networkx graph into a pytorch geometric graph
pyg_graph = from_networkx(G)

# del pyg_graph['weight']

# del pyg_graph['path']

# make it into an array TODO: Change to the actual network array that we use, for now we only have one test example.
graph_array = [pyg_graph]

with open('Data/test.p', 'wb') as handle:
    handle.write(pickle.dumps(graph_array))