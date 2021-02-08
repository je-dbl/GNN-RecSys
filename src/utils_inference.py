import pandas as pd

from dgl.data.utils import load_graphs


def read_graph(graph_path):
    """
    Read graph data from path.
    """
    graph_list, _ = load_graphs(graph_path)
    graph = graph_list[0]
    return graph


def fetch_uids(user_ids,
               ctm_id_df):
    """
    Maps the Organisation user_ids into node_ids that are used in the graph.
    """
    user_df = pd.DataFrame(user_ids, columns=['old_id'])
    user_df = user_df.merge(ctm_id_df, how='inner', left_on='old_id', right_on='CUSTOMER IDENTIFIER')
    new_uids_list = user_df.ctm_new_id.values
    if len(user_ids) != len(new_uids_list):
        print(f'{len(user_ids)-len(new_uids_list)} user ids provided had no node ids in the graph.')
    return new_uids_list


def postprocess_recs(recs,
                     pdt_id_df,
                     ctm_id_df,
                     pdt_id_type,
                     ctm_id_type, ):
    """
    Transforms node_ids for user and item into Organisation user_ids and item_ids
    (e.g.CUSTOMER IDENTIFIER and ITEM IDENTIFIER)
    """
    processed_recs = {ctm_id_df[ctm_id_df.ctm_new_id == key][ctm_id_type].item():
                          [pdt_id_df[pdt_id_df.pdt_new_id == iid][pdt_id_type].item() for iid in value_list]
                      for key, value_list in recs.items()}
    return processed_recs
