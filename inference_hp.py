import math

import numpy as np
import torch

from src.utils_data import DataLoader, DataPaths, assign_graph_features

from src.builder import (create_graph)
from src.model import ConvModel
from src.sampling import train_valid_split, generate_dataloaders
from src.metrics import get_metrics_at_k
from src.train.run import get_embeddings
from src.utils import save_txt, read_data

cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')
num_workers = 4 if cuda else 0

def inference_fn(trained_model,
                 remove,
                 fixed_params,
                 overwrite_fixed_params=False,
                 days_of_purchases=710,
                 days_of_clicks=710,
                 lifespan_of_items=710,
                 **params):
    """
    Function to run inference inside the hyperparameter loop and calculate metrics.

    Parameters
    ----------
    trained_model:
        Model trained during training of hyperparameter loop.
    remove:
        Percentage of data removed. See src.utils_data for more details.
    fixed_params:
        All parameters used during training of hyperparameter loop. See src.utils_data for more details.
    overwrite_fixed_params:
        If true, training parameters will overwritten by the parameters below. Can be useful if need to test the model
        on different parameters, e.g. that includes older clicks or purchases.
    days_of_purchases, days_of_clicks, lifespan_of_items:
        All parameters that can overwrite the training parameters. Only useful if overwrite_fixed_params is True.
    params:
        All other parameters used during training.

    Returns
    -------
    recall:
        Recall on the test set. Relevant to compare with recall computed on hyperparametrization test set (since
        parameters like 'remove' and all overwritable parameters are different)

    Saves to file
    -------------
    Metrics computed on the test set.
    """
    # Import parameters
    if isinstance(fixed_params, str):
        path = fixed_params
        fixed_params = read_data(path)
        class objectview(object):
            def __init__(self, d):
                self.__dict__ = d
        fixed_params = objectview(fixed_params)

    if 'params' in params.keys():
        # if isinstance(params['params'], str):
        path = params['params']
        params = read_data(path)

    # Initialize data
    data_paths = DataPaths()
    fixed_params.remove = remove
    if overwrite_fixed_params:
        fixed_params.days_of_purchases = days_of_purchases
        fixed_params.days_of_clicks = days_of_clicks
        fixed_params.lifespan_of_items = lifespan_of_items
    data = DataLoader(data_paths, fixed_params)

    # Get graph
    valid_graph = create_graph(
        data.graph_schema,
    )
    valid_graph = assign_graph_features(valid_graph,
                                        fixed_params,
                                        data,
                                        **params,
                                        )

    dim_dict = {'user': valid_graph.nodes['user'].data['features'].shape[1],
                'item': valid_graph.nodes['item'].data['features'].shape[1],
                'out': params['out_dim'],
                'hidden': params['hidden_dim']}

    all_sids = None
    if 'sport' in valid_graph.ntypes:
        dim_dict['sport'] = valid_graph.nodes['sport'].data['features'].shape[1]
        all_sids = np.arange(valid_graph.num_nodes('sport'))

    # get training and test ids
    (
        train_graph,
        train_eids_dict,
        valid_eids_dict,
        subtrain_uids,
        valid_uids,
        test_uids,
        all_iids,
        ground_truth_subtrain,
        ground_truth_valid,
        all_eids_dict
    ) = train_valid_split(
        valid_graph,
        data.ground_truth_test,
        fixed_params.etype,
        fixed_params.subtrain_size,
        fixed_params.valid_size,
        fixed_params.reverse_etype,
        fixed_params.train_on_clicks,
        fixed_params.remove_train_eids,
        params['clicks_sample'],
        params['purchases_sample'],
    )
    (
        edgeloader_train,
        edgeloader_valid,
        nodeloader_subtrain,
        nodeloader_valid,
        nodeloader_test
    ) = generate_dataloaders(valid_graph,
                             train_graph,
                             train_eids_dict,
                             valid_eids_dict,
                             subtrain_uids,
                             valid_uids,
                             test_uids,
                             all_iids,
                             fixed_params,
                             num_workers,
                             all_sids,
                             embedding_layer=params['embedding_layer'],
                             n_layers=params['n_layers'],
                             neg_sample_size=params['neg_sample_size'],
                             )

    num_batches_test = math.ceil((len(test_uids) + len(all_iids)) / fixed_params.node_batch_size)

    # Import model
    if isinstance(trained_model, str):
        path = trained_model
        trained_model = ConvModel(valid_graph,
                                  params['n_layers'],
                                  dim_dict,
                                  params['norm'],
                                  params['dropout'],
                                  params['aggregator_type'],
                                  fixed_params.pred,
                                  params['aggregator_hetero'],
                                  params['embedding_layer'],
                                  )
        trained_model.load_state_dict(torch.load(path, map_location=device))
    if cuda:
        trained_model = trained_model.to(device)

    trained_model.eval()
    with torch.no_grad():
        embeddings = get_embeddings(valid_graph,
                                    params['out_dim'],
                                    trained_model,
                                    nodeloader_test,
                                    num_batches_test,
                                    cuda,
                                    device,
                                    params['embedding_layer'],
                                    )

        for ground_truth in [data.ground_truth_purchase_test, data.ground_truth_test]:
            precision, recall, coverage = get_metrics_at_k(
                embeddings,
                valid_graph,
                trained_model,
                params['out_dim'],
                ground_truth,
                all_eids_dict[('user', 'buys', 'item')],
                fixed_params.k,
                True,  # Remove already bought
                cuda,
                device,
                fixed_params.pred,
                params['use_popularity'],
                params['weight_popularity'],
            )

            sentence = ("TEST Precision "
                        "{:.3f}% | Recall {:.3f}% | Coverage {:.2f}%"
                        .format(precision * 100,
                                recall * 100,
                                coverage * 100))

            print(sentence)
            save_txt(sentence, data_paths.result_filepath, mode='a')

    return recall
