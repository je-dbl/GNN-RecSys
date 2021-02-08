from src.utils import softmax
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

def create_ground_truth(users, items):
    """
    Creates a dictionary, where the keys are user ids and the values are item ids that the user actually bought.
    """
    ground_truth_arr = np.stack((np.asarray(users), np.asarray(items)), axis=1)
    ground_truth_dict = defaultdict(list)
    for key, val in ground_truth_arr:
        ground_truth_dict[key].append(val)
    return ground_truth_dict


def create_already_bought(g, bought_eids, etype='buys'):
    """
    Creates a dictionary, where the keys are user ids and the values are item ids that the user already bought.
    """
    users_train, items_train = g.find_edges(bought_eids, etype=etype)
    already_bought_arr = np.stack((np.asarray(users_train), np.asarray(items_train)), axis=1)
    already_bought_dict = defaultdict(list)
    for key, val in already_bought_arr:
        already_bought_dict[key].append(val)
    return already_bought_dict


def get_recs(g, 
             h, 
             model,
             embed_dim,
             k,
             user_ids,
             already_bought_dict,
             remove_already_bought=True,
             cuda=False,
             device=None,
             pred: str = 'cos',
             use_popularity: bool = False,
             weight_popularity=1
             ):
    """
    Computes K recommendation for all users, given hidden states, the model and what they already bought.
    """
    if cuda:  # model is already in cuda?
        model = model.to(device)
    print('Computing recommendations on {} users, for {} items'.format(len(user_ids), g.num_nodes('item')))
    recs = {}
    for user in user_ids:
        user_emb = h['user'][user]
        already_bought = already_bought_dict[user]
        user_emb_rpt = torch.cat(g.num_nodes('item')*[user_emb]).reshape(-1, embed_dim)
        
        if pred == 'cos':
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            ratings = cos(user_emb_rpt, h['item'])

        elif pred == 'nn':
            cat_embed = torch.cat((user_emb_rpt, h['item']), 1)
            ratings = model.pred_fn.layer_nn(cat_embed)

        else:
            raise KeyError(f'Prediction function {pred} not recognized.')
            
        ratings_formatted = ratings.cpu().detach().numpy().reshape(g.num_nodes('item'),)
        if use_popularity:
            softmax_ratings = softmax(ratings_formatted)
            popularity_scores = g.ndata['popularity']['item'].numpy().reshape(g.num_nodes('item'),)
            ratings_formatted = np.add(softmax_ratings, popularity_scores * weight_popularity)
        order = np.argsort(-ratings_formatted)
        if remove_already_bought:
            order = [item for item in order if item not in already_bought]
        rec = order[:k]
        recs[user] = rec
    return recs


def recs_to_metrics(recs, ground_truth_dict, g):
    """
    Given the recommendations and the ground_truth, computes precision, recall & coverage.
    """
    # precision
    k_relevant = 0
    k_total = 0
    for uid, iids in recs.items():
        k_total += len(iids)
        k_relevant += len([id_ for id_ in iids if id_ in ground_truth_dict[uid]])
    precision = k_relevant/k_total

    # recall
    k_relevant = 0
    k_total = 0
    for uid, iids in recs.items():
        k_total += len(ground_truth_dict[uid])
        k_relevant += len([id_ for id_ in ground_truth_dict[uid] if id_ in iids])
    recall = k_relevant/k_total
    
    # coverage
    nb_total = g.num_nodes('item')
    recs_flatten = [item for sublist in list(recs.values()) for item in sublist]
    nb_recommended = len(set(recs_flatten))
    coverage = nb_recommended / nb_total
    
    return precision, recall, coverage


def get_metrics_at_k(h, 
                     g,
                     model,
                     embed_dim,
                     ground_truth,
                     bought_eids,
                     k,
                     remove_already_bought=True,
                     cuda=False,
                     device=None,
                     pred='cos',
                     use_popularity=False,
                     weight_popularity=1):
    """
    Function combining all previous functions: create already_bought & ground_truth dict, get recs and compute metrics.
    """
    already_bought_dict = create_already_bought(g, bought_eids)
    users, items = ground_truth
    user_ids = np.unique(users).tolist()
    ground_truth_dict = create_ground_truth(users, items)
    recs = get_recs(g, h, model, embed_dim, k, user_ids, already_bought_dict,
                    remove_already_bought, cuda, device, pred, use_popularity, weight_popularity)
    precision, recall, coverage = recs_to_metrics(recs, ground_truth_dict, g)
    
    return precision, recall, coverage


def MRR_neg_edges(model,
                  blocks,
                  pos_g,
                  neg_g,
                  etype,
                  neg_sample_size):
    """
    (Currently not used.) Computes Mean Reciprocal Rank for the positive edge, out of all negative edges considered.

    Since it computes only on negative edges considered, it is an heuristic of the real MRR.
    However, if you put neg_sample_size as the total number of items, can be considered as MRR
    (because it will create as many edges as there are items).
    """
    input_features = blocks[0].srcdata['features']
    _, pos_score, neg_score = model(blocks,
                                    input_features,
                                    pos_g, neg_g,
                                    etype)
    neg_score = neg_score.reshape(-1, neg_sample_size)
    rankings = torch.sum(neg_score >= pos_score, dim=1) + 1
    return np.mean(1.0 / rankings.cpu().numpy())
