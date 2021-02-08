import dgl
import numpy as np


def train_valid_split(valid_graph: dgl.DGLHeteroGraph,
                      ground_truth_test,
                      etypes,
                      subtrain_size,
                      valid_size,
                      reverse_etype,
                      train_on_clicks,
                      remove_train_eids,
                      clicks_sample=1,
                      purchases_sample=1,
                      ):
    """
    Using the full graph, sample train_graph and eids of edges for train & validation, as well as nids for test.

    Process:
        - Validation
            - valid_eids are the most recent X edges of all eids of the graph (based on valid_size)
            - valid_uids and iid are the user ids and item ids associated with those edges (and together they form the
              ground_truth)
        - Training graph & eids
            - All edges and reverse edges of valid_eids are removed from the full graph.
            - train_eids are all remaining edges.
        - Sampling of training eids
            - It might be relevant to have numerous edges in the training graph to do message passing, but to
              optimize the model to give great scores only to recent interaction (to help with seasonality)
            - Thus, if purchases_sample or clicks_sample are < 1, only the most recent X edges are kept in the
              train_eids dict
            - An extra option is available to insure that no information leakage appear: remove_train_eids. If true,
              all eids in train_eids dict will be removed from the graph. (Otherwise, information leakage is still
              taken care of during EdgeDataLoader: sampled edges are removed from the computation blocks). Based on
              experience, it is best to set remove_train_eids as False.
        - Computing metrics on training set: subtrain nids
            - To compute metrics on the training set, we sample a "subtrain set". We need the ground_truth for
              the subtrain, as well as node ids for all user and items in the subtrain set.
        - Computing metrics on test set
            - We need node ids for all user and items in the test set (so we can fetch their embeddings during
              recommendations)

    """
    np.random.seed(11)

    all_eids_dict = {}
    valid_eids_dict = {}
    train_eids_dict = {}
    valid_uids_all = []
    valid_iids_all = []
    for etype in etypes:
        all_eids = np.arange(valid_graph.number_of_edges(etype))
        valid_eids = all_eids[int(len(all_eids) * (1 - valid_size)):]
        valid_uids, valid_iids = valid_graph.find_edges(valid_eids, etype=etype)
        valid_uids_all.extend(valid_uids.tolist())
        valid_iids_all.extend(valid_iids.tolist())
        all_eids_dict[etype] = all_eids
        if (etype == ('user', 'buys', 'item')) or (etype == ('user', 'clicks', 'item') and train_on_clicks):
            valid_eids_dict[etype] = valid_eids
    ground_truth_valid = (np.array(valid_uids_all), np.array(valid_iids_all))
    valid_uids = np.array(np.unique(valid_uids_all))

    # Create partial graph
    train_graph = valid_graph.clone()
    for etype in etypes:
        if (etype == ('user', 'buys', 'item')) or (etype == ('user', 'clicks', 'item') and train_on_clicks):
            train_graph.remove_edges(valid_eids_dict[etype], etype=etype)
            train_graph.remove_edges(valid_eids_dict[etype], etype=reverse_etype[etype])
            train_eids = np.arange(train_graph.number_of_edges(etype))
            train_eids_dict[etype] = train_eids

    if purchases_sample != 1:
        eids = train_eids_dict[('user', 'buys', 'item')]
        train_eids_dict[('user', 'buys', 'item')] = eids[int(len(eids) * (1 - purchases_sample)):]
        eids = valid_eids_dict[('user', 'buys', 'item')]
        valid_eids_dict[('user', 'buys', 'item')] = eids[int(len(eids) * (1 - purchases_sample)):]

    if clicks_sample != 1 and ('user', 'clicks', 'item') in train_eids_dict.keys():
        eids = train_eids_dict[('user', 'clicks', 'item')]
        train_eids_dict[('user', 'clicks', 'item')] = eids[int(len(eids) * (1 - clicks_sample)):]
        eids = valid_eids_dict[('user', 'clicks', 'item')]
        valid_eids_dict[('user', 'clicks', 'item')] = eids[int(len(eids) * (1 - clicks_sample)):]

    if remove_train_eids:
        train_graph.remove_edges(train_eids_dict[etype], etype=etype)
        train_graph.remove_edges(train_eids_dict[etype], etype=reverse_etype[etype])

    # Generate inference nodes for subtrain & ground truth for subtrain
    ## Choose the subsample of training set. For now, only users with purchases are included.
    train_uids, train_iids = valid_graph.find_edges(train_eids_dict[etypes[0]], etype=etypes[0])
    unique_train_uids = np.unique(train_uids)
    subtrain_uids = np.random.choice(unique_train_uids, int(len(unique_train_uids) * subtrain_size), replace=False)
    ## Fetch uids and iids of subtrain sample for all etypes
    subtrain_uids_all = []
    subtrain_iids_all = []
    for etype in train_eids_dict.keys():
        train_uids, train_iids = valid_graph.find_edges(train_eids_dict[etype], etype=etype)
        subtrain_eids = []
        for i in range(len(train_eids_dict[etype])):
            if train_uids[i].item() in subtrain_uids:
                subtrain_eids.append(train_eids_dict[etype][i].item())
        subtrain_uids, subtrain_iids = valid_graph.find_edges(subtrain_eids, etype=etype)
        subtrain_uids_all.extend(subtrain_uids.tolist())
        subtrain_iids_all.extend(subtrain_iids.tolist())
    ground_truth_subtrain = (np.array(subtrain_uids_all), np.array(subtrain_iids_all))
    subtrain_uids = np.array(np.unique(subtrain_uids_all))

    # Generate inference nodes for test
    test_uids, _ = ground_truth_test
    test_uids = np.unique(test_uids)
    all_iids = np.arange(valid_graph.num_nodes('item'))

    return train_graph, train_eids_dict, valid_eids_dict, subtrain_uids, valid_uids, test_uids, \
           all_iids, ground_truth_subtrain, ground_truth_valid, all_eids_dict


def generate_dataloaders(valid_graph,
                         train_graph,
                         train_eids_dict,
                         valid_eids_dict,
                         subtrain_uids,
                         valid_uids,
                         test_uids,
                         all_iids,
                         fixed_params,
                         num_workers,
                         all_sids=None,
                         embedding_layer: bool = True,
                         **params,
                         ):
    """
    Since data is large, it is fed to the model in batches. This creates batches for train, valid & test.

    Process:
        - Set up
            - Fix the number of layers. If there is an explicit embedding layer, we need 1 less layer in the blocks.
            - The sampler will generate computation blocks. Currently, only 'full' sampler is used, meaning that all
              nodes have all their neighbors, but one could specify 'partial' neighborhood to have only message passing
              with a limited number of neighbors.
            - The negative sampler generates K negative samples for all positive examples in the batch.
        - Edgeloader_train
            - All train_eids will be batched, using the training graph. Sampled edge and their reverse etype will be
              removed from computation blocks.
            - If remove_train_eids, the graph used for sampling will not have the train_eids as edges. (Thus, a
              different graph as g_sampling)
        - Edgeloader_valid
            - All valid_eids will be batched.
        - Nodeloaders
            - When computing metrics, we want to compute embeddings for all nodes of interest. Thus, we use
              a NodeDataLoader instead of an EdgeDataLoader.
            - We have a nodeloader for subtrain, validation and test.
    """
    n_layers = params['n_layers']
    if embedding_layer:
        n_layers = n_layers - 1
    if fixed_params.neighbor_sampler == 'full':
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
    elif fixed_params.neighbor_sampler == 'partial':
        sampler = dgl.dataloading.MultiLayerNeighborSampler([1, 1, 1], replace=False)
    else:
        raise KeyError('Neighbor sampler {} not recognized.'.format(fixed_params.neighbor_sampler))

    sampler_n = dgl.dataloading.negative_sampler.Uniform(
        params['neg_sample_size']
    )

    if fixed_params.remove_train_eids:
        edgeloader_train = dgl.dataloading.EdgeDataLoader(
            valid_graph,
            train_eids_dict,
            sampler,
            g_sampling=train_graph,
            negative_sampler=sampler_n,
            batch_size=fixed_params.edge_batch_size,
            shuffle=True,
            drop_last=False,  # Drop last batch if non-full
            pin_memory=True,  # Helps the transfer to GPU
            num_workers=num_workers,
        )
    else:
        edgeloader_train = dgl.dataloading.EdgeDataLoader(
            train_graph,
            train_eids_dict,
            sampler,
            exclude='reverse_types',
            reverse_etypes={'buys': 'bought-by', 'bought-by': 'buys',
                            'clicks': 'clicked-by', 'clicked-by': 'clicks'},
            negative_sampler=sampler_n,
            batch_size=fixed_params.edge_batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
        )

    edgeloader_valid = dgl.dataloading.EdgeDataLoader(
        valid_graph,
        valid_eids_dict,
        sampler,
        g_sampling=train_graph,
        negative_sampler=sampler_n,
        batch_size=fixed_params.edge_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    nodeloader_subtrain = dgl.dataloading.NodeDataLoader(
        train_graph,
        {'user': subtrain_uids, 'item': all_iids},
        sampler,
        batch_size=fixed_params.node_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    nodeloader_valid = dgl.dataloading.NodeDataLoader(
        train_graph,
        {'user': valid_uids, 'item': all_iids},
        sampler,
        batch_size=fixed_params.node_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    test_node_ids = {'user': test_uids, 'item': all_iids}
    if 'sport' in valid_graph.ntypes:
        test_node_ids['sport'] = all_sids

    nodeloader_test = dgl.dataloading.NodeDataLoader(
        valid_graph,
        test_node_ids,
        sampler,
        batch_size=fixed_params.node_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )

    return edgeloader_train, edgeloader_valid, nodeloader_subtrain, nodeloader_valid, nodeloader_test
