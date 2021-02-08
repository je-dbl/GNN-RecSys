import math
import datetime

import click
import numpy as np
import torch
from dgl.data.utils import save_graphs

from src.builder import create_graph
from src.utils_data import DataLoader, assign_graph_features
from src.utils import read_data, save_txt, save_outputs
from src.model import ConvModel, max_margin_loss
from src.sampling import train_valid_split, generate_dataloaders
from src.train.run import train_model, get_embeddings
from src.utils_vizualization import plot_train_loss
from src.metrics import (create_already_bought, create_ground_truth,
                         get_metrics_at_k, get_recs)
from src.evaluation import explore_recs, explore_sports, check_coverage
from presplit import presplit_data

from logging_config import get_logger

log = get_logger(__name__)

cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')
num_workers = 4 if cuda else 0


class TrainDataPaths:
    def __init__(self):
        self.result_filepath = 'TXT FILE WHERE TO LOG THE RESULTS .txt'
        self.sport_feat_path = 'FEATURE DATASET, SPORTS (sport names) .csv'
        self.full_interaction_path = 'INTERACTION LIST, USER-ITEM (Full dataset, not splitted between train & test).csv'
        self.item_sport_path = 'INTERACTION LIST, ITEM-SPORT .csv'
        self.user_sport_path = 'INTERACTION LIST, USER-SPORT .csv'
        self.sport_sportg_path = 'INTERACTION LIST, SPORT-SPORT .csv'
        self.item_feat_path = 'FEATURE DATASET, ITEMS .csv'
        self.user_feat_path = 'FEATURE DATASET, USERS.csv'
        self.sport_onehot_path = 'FEATURE DATASET, SPORTS (one-hot vectors) .csv'


def train_full_model(fixed_params_path,
                     visualization,
                     check_embedding,
                     remove,
                     edge_batch_size,
                     **params,):
    """
    Given the best hyperparameter combination, function to train the model on all available data.

    Files needed to run
    -------------------
    All the files in the TrainDataPaths:
        It includes all the interactions between user, sport and items, as well as features for user, sport and items.
    Fixed_params and params found in hyperparametrization:
        Those params will indicate how to train the model. Usually, they are found when running the hyperparametrization
        loop.

    Parameters
    ----------
    See click options below for details.


    Saves to files
    --------------
    trained_model with its fixed parameters and hyperparameters:
        The trained model with all parameters are saved to the folder 'models'.
    graph and ID mapping:
        When doing inference, it might be useful to import an already built graph (and the mapping that allows to
        associate node ID with personal information such as CUSTOMER IDENTIFIER or ITEM IDENTIFIER). Thus, the graph and ID mapping are saved to
        folder 'models'.
    """
    # Load parameters
    fixed_params = read_data(fixed_params_path)
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
    fixed_params = objectview(fixed_params)
    fixed_params.remove = remove
    fixed_params.subtrain_size = 0.01
    fixed_params.valid_size = 0.01
    fixed_params.edge_batch_size = edge_batch_size

    # Create full train set
    train_data_paths = TrainDataPaths()
    presplit_item_feat = read_data(train_data_paths.item_feat_path)
    full_interaction_data = read_data(train_data_paths.full_interaction_path)
    train_df, test_df = presplit_data(presplit_item_feat,
                                      full_interaction_data,
                                      num_min=3,
                                      remove_unk=True,
                                      sort=True,
                                      test_size_days=1,
                                      item_id_type='ITEM IDENTIFIER',
                                      ctm_id_type='CUSTOMER IDENTIFIER', )
    train_data_paths.train_path = train_df
    train_data_paths.test_path = test_df
    data = DataLoader(train_data_paths, fixed_params)

    # Initialize graph & features
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

    # Initialize model
    model = ConvModel(valid_graph,
                      params['n_layers'],
                      dim_dict,
                      params['norm'],
                      params['dropout'],
                      params['aggregator_type'],
                      params['pred'],
                      params['aggregator_hetero'],
                      params['embedding_layer'],
                      )
    if cuda:
        model = model.to(device)

    # Initialize dataloaders
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

    train_eids_len = 0
    valid_eids_len = 0
    for etype in train_eids_dict.keys():
        train_eids_len += len(train_eids_dict[etype])
        valid_eids_len += len(valid_eids_dict[etype])
    num_batches_train = math.ceil(train_eids_len / fixed_params.edge_batch_size)
    num_batches_subtrain = math.ceil(
        (len(subtrain_uids) + len(all_iids)) / fixed_params.node_batch_size
    )
    num_batches_val_loss = math.ceil(valid_eids_len / fixed_params.edge_batch_size)
    num_batches_val_metrics = math.ceil(
        (len(valid_uids) + len(all_iids)) / fixed_params.node_batch_size
    )
    num_batches_test = math.ceil(
        (len(test_uids) + len(all_iids)) / fixed_params.node_batch_size
    )

    # Run model
    hp_sentence = params
    hp_sentence.update(vars(fixed_params))
    hp_sentence = f'{str(hp_sentence)[1: -1]} \n'
    save_txt(f'\n \n START - Hyperparameters \n{hp_sentence}', train_data_paths.result_filepath, "a")
    trained_model, viz, best_metrics = train_model(
        model,
        fixed_params.num_epochs,
        num_batches_train,
        num_batches_val_loss,
        edgeloader_train,
        edgeloader_valid,
        max_margin_loss,
        params['delta'],
        params['neg_sample_size'],
        params['use_recency'],
        cuda,
        device,
        fixed_params.optimizer,
        params['lr'],
        get_metrics=True,
        train_graph=train_graph,
        valid_graph=valid_graph,
        nodeloader_valid=nodeloader_valid,
        nodeloader_subtrain=nodeloader_subtrain,
        k=fixed_params.k,
        out_dim=params['out_dim'],
        num_batches_val_metrics=num_batches_val_metrics,
        num_batches_subtrain=num_batches_subtrain,
        bought_eids=train_eids_dict[('user', 'buys', 'item')],
        ground_truth_subtrain=ground_truth_subtrain,
        ground_truth_valid=ground_truth_valid,
        remove_already_bought=True,
        result_filepath=train_data_paths.result_filepath,
        start_epoch=fixed_params.start_epoch,
        patience=fixed_params.patience,
        pred=params['pred'],
        use_popularity=params['use_popularity'],
        weight_popularity=params['weight_popularity'],
        remove_false_negative=fixed_params.remove_false_negative,
        embedding_layer=params['embedding_layer'],
    )

    # Get viz & metrics
    if visualization:
        plot_train_loss(hp_sentence, viz)

    # Report performance on validation set
    sentence = ("BEST VALIDATION Precision "
                "{:.3f}% | Recall {:.3f}% | Coverage {:.2f}%"
                .format(best_metrics['precision'] * 100,
                        best_metrics['recall'] * 100,
                        best_metrics['coverage'] * 100))

    log.info(sentence)
    save_txt(sentence, train_data_paths.result_filepath, mode='a')

    # Report performance on test set
    log.debug('Test metrics start ...')
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
                params['pred'],
                params['use_popularity'],
                params['weight_popularity'],
            )

            sentence = ("TEST Precision "
                        "{:.3f}% | Recall {:.3f}% | Coverage {:.2f}%"
                        .format(precision * 100,
                                recall * 100,
                                coverage * 100))
            log.info(sentence)
            save_txt(sentence, train_data_paths.result_filepath, mode='a')

    if check_embedding:
        trained_model.eval()
        with torch.no_grad():
            log.debug('ANALYSIS OF RECOMMENDATIONS')
            if 'sport' in train_graph.ntypes:
                result_sport = explore_sports(embeddings,
                                              data.sport_feat_df,
                                              data.spt_id,
                                              fixed_params.num_choices)

                save_txt(result_sport, train_data_paths.result_filepath, mode='a')

            already_bought_dict = create_already_bought(valid_graph,
                                                        all_eids_dict[('user', 'buys', 'item')],
                                                        )
            already_clicked_dict = None
            if fixed_params.discern_clicks:
                already_clicked_dict = create_already_bought(valid_graph,
                                                             all_eids_dict[('user', 'clicks', 'item')],
                                                             etype='clicks',
                                                             )

            users, items = data.ground_truth_test
            ground_truth_dict = create_ground_truth(users, items)
            user_ids = np.unique(users).tolist()
            recs = get_recs(valid_graph,
                            embeddings,
                            trained_model,
                            params['out_dim'],
                            fixed_params.k,
                            user_ids,
                            already_bought_dict,
                            remove_already_bought=True,
                            pred=params['pred'],
                            use_popularity=params['use_popularity'],
                            weight_popularity=params['weight_popularity'])

            users, items = data.ground_truth_purchase_test
            ground_truth_purchase_dict = create_ground_truth(users, items)
            explore_recs(recs,
                         already_bought_dict,
                         already_clicked_dict,
                         ground_truth_dict,
                         ground_truth_purchase_dict,
                         data.item_feat_df,
                         fixed_params.num_choices,
                         data.pdt_id,
                         fixed_params.item_id_type,
                         train_data_paths.result_filepath)

            if fixed_params.item_id_type == 'SPECIFIC ITEM IDENTIFIER':
                coverage_metrics = check_coverage(data.user_item_train,
                                                  data.item_feat_df,
                                                  data.pdt_id,
                                                  recs)

                sentence = (
                    "COVERAGE \n|| All transactions : "
                    "Generic {:.1f}% | Junior {:.1f}% | Male {:.1f}% | Female {:.1f}% | Eco {:.1f}% "
                    "\n|| Recommendations : "
                    "Generic {:.1f}% | Junior {:.1f}% | Male {:.1f}% | Female {:.1f} | Eco {:.1f}%%"
                        .format(
                        coverage_metrics['generic_mean_whole'] * 100,
                        coverage_metrics['junior_mean_whole'] * 100,
                        coverage_metrics['male_mean_whole'] * 100,
                        coverage_metrics['female_mean_whole'] * 100,
                        coverage_metrics['eco_mean_whole'] * 100,
                        coverage_metrics['generic_mean_recs'] * 100,
                        coverage_metrics['junior_mean_recs'] * 100,
                        coverage_metrics['male_mean_recs'] * 100,
                        coverage_metrics['female_mean_recs'] * 100,
                        coverage_metrics['eco_mean_recs'] * 100,
                    )
                )
                log.info(sentence)
                save_txt(sentence, train_data_paths.result_filepath, mode='a')

        save_outputs(
            {
                'embeddings': embeddings,
                'already_bought': already_bought_dict,
                'already_clicked': already_bought_dict,
                'ground_truth': ground_truth_dict,
                'recs': recs,
            },
            'outputs/'
        )

    # Save model
    date = str(datetime.datetime.now())[:-10].replace(' ', '')
    torch.save(trained_model.state_dict(), f'models/FULL_Recall_{recall * 100:.2f}_{date}.pth')
    # Save all necessary params
    save_outputs(
        {
            f'{date}_params': params,
            f'{date}_fixed_params': vars(fixed_params),
        },
        'models/'
    )
    print("Saved model & parameters to disk.")

    # Save graph & ID mapping
    save_graphs(f'models/{date}_graph.bin', [valid_graph])
    save_outputs(
        {
            f'{date}_ctm_id': data.ctm_id,
            f'{date}_pdt_id': data.pdt_id,
        },
        'models/'
    )
    print("Saved graph & ID mapping to disk.")


@click.command()
@click.option('--fixed_params_path', default='fixed_params.pkl',
              help='Path where the fixed parameters used in the hyperparametrization were saved.')
@click.option('--params_path', default='params.pkl',
              help='Path where the optimal hyperparameters found in the hyperparametrization were saved.')
@click.option('-viz', '--visualization', count=True, help='Visualize result')
@click.option('--check_embedding', count=True, help='Explore embedding result')
@click.option('--remove', default=.99, help='Percentage of users to remove from train set. Ideally,'
                                            ' remove would be 0. However, higher "remove" accelerates training.')
@click.option('--edge_batch_size', default=2048, help='Number of edges in a train / validation batch')
def main(fixed_params_path, params_path, visualization, check_embedding, remove, edge_batch_size):
    params = read_data(params_path)
    params.pop('remove', None)
    params.pop('edge_batch_size', None)
    train_full_model(fixed_params_path=fixed_params_path,
                     visualization=visualization,
                     check_embedding=check_embedding,
                     remove=remove,
                     edge_batch_size=edge_batch_size,
                     **params)

if __name__ == '__main__':
    main()
