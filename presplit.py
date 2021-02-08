from datetime import datetime, timedelta

import numpy as np

from logging_config import get_logger

logger = get_logger(__file__)


def presplit_data(item_feature_data,
                  user_item_interaction_data,
                  num_min=3,
                  remove_unk=True,
                  sort=True,
                  test_size_days=14,
                  item_id_type='ITEM IDENTIFIER',
                  ctm_id_type='CUSTOMER IDENTIFIER'):
    """
    Split data into train and test set.

    Parameters
    ----------
    num_min:
        Minimal number of interactions (transactions or clicks) for a customer to be included in the dataset
        (interactions can be both in train and test sets)
    remove_unk:
        Remove items in the interaction set that are not in the item features set, e.g. "items" that are services
        like skate sharpening
    sort:
        Sort the dataset by date before splitting in train/test set,  thus having a test set that is succeeding
        the train set
    test_size_days:
        Number of days that should be in the test set. The rest will be in the training set.
    ctm_id_type:
        Unique identifier for the customers.
    item_id_type:
        Unique identifier for the items.

    Returns
    -------
    train_set:
        Pandas dataframe of all training interactions.
    test_set:
        Pandas dataframe of all testing interactions.
    """

    np.random.seed(11)

    if num_min > 0:
        user_item_interaction_data = user_item_interaction_data[
            user_item_interaction_data[ctm_id_type].map(
                user_item_interaction_data[ctm_id_type].value_counts()
            ) >= num_min
        ]

    if remove_unk:
        known_items = item_feature_data[item_id_type].unique().tolist()
        user_item_interaction_data = user_item_interaction_data[user_item_interaction_data[item_id_type].isin(known_items)]

    if sort:
        user_item_interaction_data.sort_values(by=['hit_timestamp'],
                                               axis=0,
                                               inplace=True)
        # Split into train & test sets
        most_recent_date = datetime.strptime(max(user_item_interaction_data.hit_date), '%Y-%m-%d')
        limit_date = datetime.strftime(
            (most_recent_date - timedelta(days=int(test_size_days))),
            format='%Y-%m-%d'
        )
        train_set = user_item_interaction_data[user_item_interaction_data['hit_date'] <= limit_date]
        test_set = user_item_interaction_data[user_item_interaction_data['hit_date'] > limit_date]

    else:
        most_recent_date = datetime.strptime(max(user_item_interaction_data.hit_date), '%Y-%m-%d')
        oldest_date = datetime.strptime(min(user_item_interaction_data.hit_date), '%Y-%m-%d')
        total_days = timedelta(days=(most_recent_date - oldest_date))  # To be tested
        test_size = test_size_days / total_days
        test_set = user_item_interaction_data.sample(frac=test_size, random_state=200)
        train_set = user_item_interaction_data.drop(test_set.index)

    # Keep only users in train set
    ctm_list = train_set[ctm_id_type].unique()
    test_set = test_set[test_set[ctm_id_type].isin(ctm_list)]
    return train_set, test_set
