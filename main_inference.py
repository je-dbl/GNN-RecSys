import math

import click
import dgl
import numpy as np
import torch

from src.builder import create_graph
from src.model import ConvModel
from src.utils_data import DataPaths, DataLoader, FixedParameters, assign_graph_features
from src.utils_inference import read_graph, fetch_uids, postprocess_recs
from src.train.run import get_embeddings
from src.metrics import get_recs, create_already_bought
from src.utils import read_data

cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')
num_workers = 4 if cuda else 0

def inference_ondemand(user_ids,  # List or 'all'
                       use_saved_graph: bool,
                       trained_model_path: str,
                       use_saved_already_bought: bool,
                       graph_path=None,
                       ctm_id_path=None,
                       pdt_id_path=None,
                       already_bought_path=None,
                       k=10,
                       remove=.99,
                       **params,
                       ):
    """
    Given a fully trained model, return recommendations specific to each user.

    Files needed to run
    -------------------
    Params used when training the model:
        Those params will indicate how to run inference on the model. Usually, they are outputted during training
        (and hyperparametrization).
    If using a saved already bought dict:
        The already bought dict: the dict includes all previous purchases of all user ids for which recommendations
                                 were requested. If not using a saved dict, it will be created using the graph.
                                 Using a saved already bought dict is not necessary, but might make the inference
                                 process faster.
    A) If using a saved graph:
        The saved graph: the graph that must include all user ids for which recommendations were requested. Usually,
                         it is outputted during training. It could also be created by another independent function.
        ID mapping: ctm_id and pdt_id mapping that allows to associate real-world information, e.g. item and customer
        identifier, to actual nodes in the graph. They are usually saved when generating a graph.
    B) If not using a saved graph:
        The graph will be generated on demand, using all the files in DataPaths of src.utils_data. All those files will
        be needed.

    Parameters
    ----------
    See click options below for details.

    Returns
    -------
    Recommendations for all user ids.

    """
    # Load & preprocess data
    ## Graph
    if use_saved_graph:
        graph = read_graph(graph_path)
        ctm_id_df = read_data(ctm_id_path)
        pdt_id_df = read_data(pdt_id_path)
    else:
        # Create graph
        data_paths = DataPaths()
        fixed_params = FixedParameters(num_epochs=0, start_epoch=0,  # Not used (only used in training)
                                       patience=0, edge_batch_size=0,  # Not used (only used in training)
                                       remove=remove, item_id_type=params['item_id_type'],
                                       duplicates=params['duplicates'])
        data = DataLoader(data_paths, fixed_params)
        ctm_id_df = data.ctm_id
        pdt_id_df = data.pdt_id

        graph = create_graph(
            data.graph_schema,
        )
        graph = assign_graph_features(graph,
                                      fixed_params,
                                      data,
                                      **params,
                                      )
    ## Preprocess: fetch right user ids
    if user_ids[0] == 'all':
        test_uids = np.arange(graph.num_nodes('user'))
    else:
        test_uids = fetch_uids(user_ids,
                               ctm_id_df)
    ## Remove already bought
    if use_saved_already_bought:
        already_bought_dict = read_data(already_bought_path)
    else:
        bought_eids = graph.out_edges(u=test_uids, form='eid', etype='buys')
        already_bought_dict = create_already_bought(graph, bought_eids)

    # Load model
    dim_dict = {'user': graph.nodes['user'].data['features'].shape[1],
                'item': graph.nodes['item'].data['features'].shape[1],
                'out': params['out_dim'],
                'hidden': params['hidden_dim']}
    if 'sport' in graph.ntypes:
        dim_dict['sport'] = graph.nodes['sport'].data['features'].shape[1]
    trained_model = ConvModel(
        graph,
        params['n_layers'],
        dim_dict,
        params['norm'],
        params['dropout'],
        params['aggregator_type'],
        params['pred'],
        params['aggregator_hetero'],
        params['embedding_layer'],
    )
    trained_model.load_state_dict(torch.load(trained_model_path, map_location=device))
    if cuda:
        trained_model = trained_model.to(device)

    # Create dataloader
    all_iids = np.arange(graph.num_nodes('item'))
    test_node_ids = {'user': test_uids, 'item': all_iids}
    n_layers = params['n_layers']
    if params['embedding_layer']:
        n_layers = n_layers - 1
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
    nodeloader_test = dgl.dataloading.NodeDataLoader(
        graph,
        test_node_ids,
        sampler,
        batch_size=128,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )
    num_batches_test = math.ceil((len(test_uids) + len(all_iids)) / 128)

    # Fetch recs
    trained_model.eval()
    with torch.no_grad():
        embeddings = get_embeddings(graph,
                                    params['out_dim'],
                                    trained_model,
                                    nodeloader_test,
                                    num_batches_test,
                                    cuda,
                                    device,
                                    params['embedding_layer'],
                                    )
        recs = get_recs(graph,
                        embeddings,
                        trained_model,
                        params['out_dim'],
                        k,
                        test_uids,
                        already_bought_dict,
                        remove_already_bought=True,
                        cuda=cuda,
                        device=device,
                        pred=params['pred'],
                        use_popularity=params['use_popularity'],
                        weight_popularity=params['weight_popularity']
                        )

        # Postprocess: user & item ids
        processed_recs = postprocess_recs(recs,
                                          pdt_id_df,
                                          ctm_id_df,
                                          params['item_id_type'],
                                          params['ctm_id_type'])
        print(processed_recs)
        return processed_recs



@click.command()
@click.option('--params_path', default='params.pkl',
              help='Path where the optimal hyperparameters found in the hyperparametrization were saved.')
@click.option('--user_ids', multiple=True, default=['all'],
              help="IDs of users for which to generate recommendations. Either list of user ids, or 'all'.")
@click.option('--use_saved_graph', count=True,
              help='If true, will use graph that was saved on disk. Need to import ID mapping for users & items.')
@click.option('--trained_model_path', default='model.pth',
              help='Path where fully trained model is saved.')
@click.option('--use_saved_already_bought', count=True,
              help='If true, will use already bought dict that was saved on disk.')
@click.option('--graph_path', default='graph.bin',
              help='Path where the graph was saved. Mandatory if use_saved_graph is True.')
@click.option('--ctm_id_path', default='ctm_id.pkl',
              help='Path where the mapping for customer was save. Mandatory if use_saved_graph is True.')
@click.option('--pdt_id_path', default='pdt_id.pkl',
              help='Path where the mapping for items was save. Mandatory if use_saved_graph is True.')
@click.option('--already_bought_path', default='already_bought.pkl',
              help='Path where the already bought dict was saved. Mandatory if use_saved_already_bought is True.')
@click.option('--k', default=10,
              help='Number of recs to generate for each user.')
@click.option('--remove', default=.99,
              help='Percentage of users to remove from graph if used_saved_graph = True. If more than 0, user_ids might'
                   ' not be in the graph. However, higher "remove" allows for faster inference.')
def main(params_path, user_ids, use_saved_graph, trained_model_path,
         use_saved_already_bought, graph_path, ctm_id_path, pdt_id_path,
         already_bought_path, k, remove):
    params = read_data(params_path)
    params.pop('k', None)
    params.pop('remove', None)


    inference_ondemand(user_ids=user_ids,  # List or 'all'
                       use_saved_graph=use_saved_graph,
                       trained_model_path=trained_model_path,
                       use_saved_already_bought=use_saved_already_bought,
                       graph_path=graph_path,
                       ctm_id_path=ctm_id_path,
                       pdt_id_path=pdt_id_path,
                       already_bought_path=already_bought_path,
                       k=k,
                       remove=remove,
                       **params,
                       )


if __name__ == '__main__':
    main()


