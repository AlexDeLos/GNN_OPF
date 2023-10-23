import math
import pandapower.plotting as ppl
import torch as th
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import DeepGraphInfomax
import tqdm


# return a torch_geometric.data.Data object for each instance
def create_data_instance(graph, y_bus, missing, volt):
    g = ppl.create_nxgraph(graph, include_trafos=True)
    # https://pandapower.readthedocs.io/en/latest/elements/gen.html
    gen = graph.gen[['bus', 'p_mw', 'vm_pu']]
    gen.rename(columns={'p_mw': 'p_mw_gen'}, inplace=True)
    gen['is_gen'] = 1
    gen.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/sgen.html
    # Note: multiple static generators can be attached to 1 bus!
    sgen = graph.sgen[['bus', 'p_mw', 'q_mvar']]
    sgen.rename(columns={'p_mw': 'p_mw_sgen'}, inplace=True)
    sgen.rename(columns={'q_mvar': 'q_mvar_sgen'}, inplace=True)
    sgen = sgen.groupby('bus')[['p_mw_sgen', 'q_mvar_sgen']].sum()  # Already resets index
    sgen['is_sgen'] = 1

    # https://pandapower.readthedocs.io/en/latest/elements/load.html
    load = graph.load[['bus', 'p_mw', 'q_mvar']]
    load.rename(columns={'p_mw': 'p_mw_load'}, inplace=True)
    load.rename(columns={'q_mvar': 'q_mvar_load'}, inplace=True)
    load['is_load'] = 1
    load.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/ext_grid.html
    ext = graph.ext_grid[['bus', 'vm_pu', 'va_degree']]
    ext.rename(columns={'vm_pu': 'vm_pu_ext'}, inplace=True)
    ext['is_ext'] = 1
    ext.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/shunt.html
    shunt = graph.shunt[['bus', 'q_mvar']]
    shunt.rename(columns={'q_mvar': 'q_mvar_shunt'}, inplace=True)
    shunt.set_index('bus', inplace=True)

    # https://pandapower.readthedocs.io/en/latest/elements/bus.html
    node_feat = graph.bus[['vn_kv']]

    # make sure all nodes (bus, gen, load) have the same number of features (namely the union of all features)
    node_feat = node_feat.merge(gen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(sgen, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(load, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(ext, left_index=True, right_index=True, how='outer')
    node_feat = node_feat.merge(shunt, left_index=True, right_index=True, how='outer')

    # fill missing feature values with 0
    node_feat.fillna(0.0, inplace=True)
    node_feat['vm_pu'] = node_feat['vm_pu'] + node_feat['vm_pu_ext']
    node_feat['p_mw'] = node_feat['p_mw_load'] - node_feat['p_mw_gen'] - node_feat['p_mw_sgen']
    node_feat['q_mvar'] = node_feat['q_mvar_load'] + node_feat['q_mvar_shunt'] - node_feat['q_mvar_sgen']

    # static generators are modeled as loads in PandaPower
    node_feat['is_load'] = (node_feat['is_sgen'] != 0) | (node_feat['is_load'] != 0)

    del node_feat['vm_pu_ext']
    del node_feat['p_mw_gen']
    del node_feat['p_mw_sgen']
    del node_feat['p_mw_load']
    del node_feat['q_mvar_load']
    del node_feat['q_mvar_sgen']
    del node_feat['q_mvar_shunt']
    del node_feat['is_sgen']

    # remove duplicate columns/indices
    node_feat = node_feat[~node_feat.index.duplicated(keep='first')]
    node_feat['is_none'] = (node_feat['is_gen'] == 0) & (node_feat['is_load'] == 0) & (node_feat['is_ext'] == 0)
    node_feat['is_none'] = node_feat['is_none'].astype(float)
    node_feat = node_feat[['is_load', 'is_gen', 'is_ext', 'is_none', 'p_mw', 'q_mvar', 'va_degree', 'vm_pu']]
    zero_check = node_feat[(node_feat['is_load'] == 0) & (node_feat['is_gen'] == 0) & (node_feat['is_ext'] == 0) & (
            node_feat['is_none'] == 0)]

    if not zero_check.empty:
        print("zero check failed")
        print(node_feat)
        print("zero check results")
        print(zero_check)
        quit()

    for node in node_feat.itertuples():
        # set each node features
        g.nodes[node.Index]['x'] = [float(node.is_load),
                                    float(node.is_gen),
                                    float(node.is_ext),
                                    float(node.is_none),
                                    float(node.p_mw),
                                    float(node.q_mvar),
                                    float(node.va_degree),
                                    float(node.vm_pu)]

        if missing:
            if node.is_load or node.is_none:
                g.nodes[node.Index]['y'] = [float(y_bus['vm_pu'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
            if node.is_gen and not node.is_load:
                g.nodes[node.Index]['y'] = [float(y_bus['q_mvar'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
            if node.is_ext:
                g.nodes[node.Index]['y'] = [float(y_bus['vm_pu'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
        elif volt:
            g.nodes[node.Index]['y'] = [float(y_bus['vm_pu'][node.Index]),
                            float(y_bus['va_degree'][node.Index])]
             
        else:
            g.nodes[node.Index]['y'] = [float(y_bus['p_mw'][node.Index]),
                                        float(y_bus['q_mvar'][node.Index]),
                                        float(y_bus['vm_pu'][node.Index]),
                                        float(y_bus['va_degree'][node.Index])]

    for edges in graph.line.itertuples():
        g.edges[edges.from_bus, edges.to_bus, ('line', edges.Index)]['edge_attr'] = [float(edges.r_ohm_per_km),
                                                                                     float(edges.x_ohm_per_km),
                                                                                     float(edges.length_km)]

    for trafos in graph.trafo.itertuples():
        g.edges[trafos.lv_bus, trafos.hv_bus, ('trafo', trafos.Index)]['edge_attr'] = [float(trafos.vkr_percent / (trafos.sn_mva / (trafos.vn_lv_kv * math.sqrt(3)))),
                                                                                       float(math.sqrt((trafos.vk_percent ** 2) - (trafos.vkr_percent) ** 2)) / (trafos.sn_mva / (trafos.vn_lv_kv * math.sqrt(3))),
                                                                                       1.0]

    return from_networkx(g)


def normalize_data(train, val, test, standard_normalizaton=True):
    combined_x = th.cat([data.x for data in train + val + test], dim=0)
    combined_y = th.cat([data.y for data in train + val + test], dim=0)
    combined_edge_attr = th.cat([data.edge_attr for data in train + val + test], dim=0)

    epsilon = 1e-7  # to avoid division by zero

    # Standard normalization between -1 and 1
    if standard_normalizaton:

        mean_x = th.mean(combined_x, dim=0)
        std_x = th.std(combined_x, dim=0)

        mean_y = th.mean(combined_y, dim=0)
        std_y = th.std(combined_y, dim=0)

        mean_edge_attr = th.mean(combined_edge_attr, dim=0)
        std_edge_attr = th.std(combined_edge_attr, dim=0)

        # normalize data
        for data in train + val + test:
            data.x = (data.x - mean_x) / (std_x + epsilon)
            data.y = (data.y - mean_y) / (std_y + epsilon)
            data.edge_attr = (data.edge_attr - mean_edge_attr) / (std_edge_attr + epsilon)

    else:  # Use min max normalization to normalize data between 0 and 1
        # https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)

        # find min value and max for all columns
        min_x = th.min(combined_x,
                       dim=0).values
        max_x = th.max(combined_x, dim=0).values

        min_y = th.min(combined_y, dim=0).values  
        max_y = th.max(combined_y, dim=0).values

        min_edge_attr = th.min(combined_edge_attr,
                               dim=0).values
        max_edge_attr = th.max(combined_edge_attr,
                               dim=0).values
        # normalize data
        for data in train + val + test:
            data.x = (data.x - min_x) / (max_x - min_x + epsilon)
            data.y = (data.y - min_y) / (max_y - min_y + epsilon)
            data.edge_attr = (data.edge_attr - min_edge_attr) / (max_edge_attr - min_edge_attr + epsilon)

    return train, val, test


# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.DeepGraphInfomax.html
# examples: 
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/infomax_transductive.py
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/infomax_inductive.py
# paper: https://arxiv.org/pdf/1809.10341.pdf
def pretrain(encoder_class, input_dim, output_dim, edge_attr_dim, train_dataloader):
    device = 'cuda' if th.cuda.is_available() else 'cpu'

    def corruption(data):
        data.x = data.x[th.randperm(data.x.size(0))]
        return data

    def train():
        model.train()
        total_loss = 0 
        total_examples = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = batch.to(device)
            pos_z, neg_z, summary = model(batch)
            loss = model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pos_z.size(0)
            total_examples += pos_z.size(0)
            
        return total_loss / total_examples
    
    model = DeepGraphInfomax(
        hidden_channels=output_dim, 
        encoder=encoder_class(
            input_dim, 
            output_dim, 
            edge_attr_dim,
        ),
        summary=lambda z, *args, **kwargs: th.sigmoid(z.mean(dim=0)),
        corruption=corruption
        ).to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in tqdm.tqdm(range(150)):
        loss = train()
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch} loss: {train():.3f}")

    return model.encoder