import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import save_txt


def get_item_by_id(iid: int,
                   pdt_id: pd.DataFrame,
                   item_feat: pd.DataFrame,
                   item_id_type: str):
    """
    Fetch information about the item, given its node_id.

    The info need to be available in the item features dataset.
    """
    # fetch old iid
    old_iid = pdt_id[item_id_type][pdt_id.pdt_new_id == iid].item()
    # fetch info
    info1 = item_feat.info1[item_feat[item_id_type] == old_iid].tolist()[0]
    info2 = item_feat.info2[item_feat[item_id_type] == old_iid].tolist()[0]
    info3 = item_feat.info3[item_feat[item_id_type] == old_iid].tolist()[0]
    return info1, info2, info3


def fetch_recs_for_users(user,
                         user_dict,
                         pdt_id,
                         item_feat_df,
                         item_id_type,
                         result_filepath,
                         ground_truth_purchase_dict=None):
    """
    For all items in a dict (of recs, or already_bought, or ground_truth), fetch information.

    """
    for iid in user_dict[user]:
        try:
            info1, info2, info3 = get_item_by_id(iid, pdt_id, item_feat_df, item_id_type)
            sentence = info1 + ', ' + info2 + info3
            if ground_truth_purchase_dict is not None:
                if iid in ground_truth_purchase_dict[user]:
                    count_purchases = len([item for item in ground_truth_purchase_dict[user] if item == iid])
                    sentence += f' ----- BOUGHT {count_purchases} TIME(S)'
        except:
            sentence = 'No name'
        save_txt(sentence, result_filepath, mode='a')


def explore_recs(recs: dict,
                 already_bought_dict: dict,
                 already_clicked_dict,
                 ground_truth_dict: dict,
                 ground_truth_purchase_dict: dict,
                 item_feat_df: pd.DataFrame,
                 num_choices: int,
                 pdt_id: pd.DataFrame,
                 item_id_type: str,
                 result_filepath: str):
    """
    For a random sample of users, fetch information about what items were clicked/bought, recommended and ground truth.

    Users with only 1 previous click or purchase are explored at the end.
    """
    choices = random.sample(recs.keys(), num_choices)

    for user in choices:
        save_txt('\nCustomer bought', result_filepath, mode='a')
        try:
            fetch_recs_for_users(user,
                                 already_bought_dict,
                                 pdt_id,
                                 item_feat_df,
                                 item_id_type,
                                 result_filepath)
        except:
            save_txt('Nothing', result_filepath, mode='a')

        save_txt('\nCustomer clicked on', result_filepath, mode='a')
        try:
            fetch_recs_for_users(user,
                                 already_clicked_dict,
                                 pdt_id,
                                 item_feat_df,
                                 item_id_type,
                                 result_filepath)
        except:
            save_txt('No click data', result_filepath, mode='a')

        save_txt('\nGot recommended', result_filepath, mode='a')
        fetch_recs_for_users(user,
                             recs,
                             pdt_id,
                             item_feat_df,
                             item_id_type,
                             result_filepath)

        save_txt('\nGround truth', result_filepath, mode='a')
        fetch_recs_for_users(user,
                             ground_truth_dict,
                             pdt_id,
                             item_feat_df,
                             item_id_type,
                             result_filepath,
                             ground_truth_purchase_dict)

    # user with 1 item
    choices = random.sample([uid for uid, v in already_bought_dict.items() if len(v) == 1 and uid in recs.keys()], 2)
    for user in choices:
        save_txt('\nCustomer bought', result_filepath, mode='a')
        try:
            fetch_recs_for_users(user,
                                 already_bought_dict,
                                 pdt_id,
                                 item_feat_df,
                                 item_id_type,
                                 result_filepath)
        except:
            save_txt('Nothing', result_filepath, mode='a')

        save_txt('\nCustomer clicked on', result_filepath, mode='a')
        try:
            fetch_recs_for_users(user,
                                 already_clicked_dict,
                                 pdt_id,
                                 item_feat_df,
                                 item_id_type,
                                 result_filepath)
        except:
            save_txt('No click data', result_filepath, mode='a')

        save_txt('\nGot recommended', result_filepath, mode='a')
        fetch_recs_for_users(user,
                             recs,
                             pdt_id,
                             item_feat_df,
                             item_id_type,
                             result_filepath)

        save_txt('\nGround truth', result_filepath, mode='a')
        fetch_recs_for_users(user,
                             ground_truth_dict,
                             pdt_id,
                             item_feat_df,
                             item_id_type,
                             result_filepath,
                             ground_truth_purchase_dict)


def explore_sports(h,
                   sport_feat_df: pd.DataFrame,
                   spt_id: pd.DataFrame,
                   num_choices: int,
                   ):
    """
    For a random sample of sport, fetch name of 5 most similar sports.
    """
    sport_h = h['sport']
    sim_matrix = cosine_similarity(sport_h.detach().cpu())
    choices = random.sample(range(sport_h.shape[0]), num_choices)
    sentence = ''
    for sid in choices:
        # fetch name of sport id
        try:
            old_sid = spt_id.sport_id[spt_id.spt_new_id == sid].item()
            chosen_name = sport_feat_df.sport_label[sport_feat_df.sport_id == old_sid].item()
        except:
            chosen_name = 'N/A'
        # fetch most similar sports
        top = np.argpartition(sim_matrix[sid], -5)[-5:]
        top_list = spt_id.sport_id[spt_id.spt_new_id.isin(top.tolist())].tolist()
        top_names = sport_feat_df.sport_label[sport_feat_df.sport_id.isin(top_list)].unique()
        sentence += 'For sport {}, top similar sports are {} \n'.format(chosen_name, top_names)
    return sentence


def check_coverage(user_item_interaction,
                   item_feat_df,
                   pdt_id,
                   recs):
    """
    Check the repartition of types of items in the purchases vs recommendations (generic vs female vs male vs junior).

    Also checks repartition of eco-design products in purchases vs recommendations.
    """
    coverage_metrics = {}

    # remove all 'unknown' items
    known_items = item_feat_df.item_identifier.unique().tolist()
    user_item_interaction = user_item_interaction[user_item_interaction.item_identifier.isin(known_items)]

    # count number of types in original dataset
    df = user_item_interaction.merge(item_feat_df,
                                     how='left',
                                     on='ITEM IDENTIFIER')
    df['is_generic'] = (df.is_junior + df.is_male + df.is_female).astype(bool) * -1 + 1

    coverage_metrics['generic_mean_whole'] = df.is_generic.mean()
    coverage_metrics['junior_mean_whole'] = df.is_junior.mean()
    coverage_metrics['male_mean_whole'] = df.is_male.mean()
    coverage_metrics['female_mean_whole'] = df.is_female.mean()
    coverage_metrics['eco_mean_whole'] = df.eco_design.mean()

    # count in 'recs'
    recs_df = pd.DataFrame(recs.items())
    recs_df.columns = ['uid', 'iid']
    recs_df = recs_df.explode('iid')
    recs_df = recs_df.merge(pdt_id,
                            how='left',
                            left_on='iid',
                            right_on='pdt_new_id')
    recs_df = recs_df.merge(item_feat_df,
                            how='left',
                            on='ITEM IDENTIFIER')

    recs_df['is_generic'] = (recs_df.is_junior + recs_df.is_male + recs_df.is_female).astype(bool) * -1 + 1

    coverage_metrics['generic_mean_recs'] = recs_df.is_generic.mean()
    coverage_metrics['junior_mean_recs'] = recs_df.is_junior.mean()
    coverage_metrics['male_mean_recs'] = recs_df.is_male.mean()
    coverage_metrics['female_mean_recs'] = recs_df.is_female.mean()
    coverage_metrics['eco_mean_recs'] = recs_df.eco_design.mean()

    return coverage_metrics
