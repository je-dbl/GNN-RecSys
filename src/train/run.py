from datetime import timedelta
import time

import dgl
import torch

from src.metrics import get_metrics_at_k
from src.utils import save_txt


def train_model(model,
                num_epochs,
                num_batches_train,
                num_batches_val_loss,
                edgeloader_train,
                edgeloader_valid,
                loss_fn,
                delta,
                neg_sample_size,
                use_recency=False,
                cuda=False,
                device=None,
                optimizer=torch.optim.Adam,
                lr=0.001,
                get_metrics=False,
                train_graph=None,
                valid_graph=None,
                nodeloader_valid=None,
                nodeloader_subtrain=None,
                k=None,
                out_dim=None,
                num_batches_val_metrics=None,
                num_batches_subtrain=None,
                bought_eids=None,
                ground_truth_subtrain=None,
                ground_truth_valid=None,
                remove_already_bought=True,
                result_filepath=None,
                start_epoch=0,
                patience=5,
                pred=None,
                use_popularity=False,
                weight_popularity=1,
                remove_false_negative=False,
                embedding_layer=True,
                ):
    """
    Main function to train a GNN, using max margin loss on positive and negative examples.

    Process:
        - A full training epoch
            - Batch by batch. 1 batch is composed of multiple computational blocks, required to compute embeddings
              for all the nodes related to the edges in the batch.
            - Input the initial features. Compute the embeddings & the positive and negative scores
            - Also compute other considerations for the loss function: negative mask, recency scores
            - Loss is returned, then backward, then step.
            - Metrics are computed on the subtraining set (using nodeloader)
        - Validation set
            - Loss is computed (in model.eval() mode) for validation edge for early stopping purposes
            - Also, metrics are computed on the validation set (using nodeloader)
        - Logging & early stopping
            - Everything is logged, best metrics are saved.
            - Using the patience parameter, early stopping is applied when val_loss stops going down.
    """
    model.train_loss_list = []
    model.train_precision_list = []
    model.train_recall_list = []
    model.train_coverage_list = []
    model.val_loss_list = []
    model.val_precision_list = []
    model.val_recall_list = []
    model.val_coverage_list = []
    best_metrics = {}  # For visualization
    max_metric = -0.1
    patience_counter = 0  # For early stopping
    min_loss = 1.1

    opt = optimizer(model.parameters(),
                    lr=lr)

    # TRAINING
    print('Starting training.')
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        print('TRAINING LOSS')
        model.train()  # Because if not, after eval, dropout would be still be inactive
        i = 0
        total_loss = 0
        for _, pos_g, neg_g, blocks in edgeloader_train:
            opt.zero_grad()

            # Negative mask
            negative_mask = {}
            if remove_false_negative:
                nids = neg_g.ndata[dgl.NID]
                for etype in pos_g.canonical_etypes:
                    neg_src, neg_dst = neg_g.edges(etype=etype)
                    neg_src = nids[etype[0]][neg_src]
                    neg_dst = nids[etype[2]][neg_dst]
                    negative_mask_tensor = valid_graph.has_edges_between(neg_src, neg_dst, etype=etype)
                    negative_mask[etype] = negative_mask_tensor.type(torch.float)
                    if cuda:
                        negative_mask[etype] = negative_mask[etype].to(device)
            if cuda:
                blocks = [b.to(device) for b in blocks]
                pos_g = pos_g.to(device)
                neg_g = neg_g.to(device)

            i += 1
            if i % 10 == 0:
                print("Edge batch {} out of {}".format(i, num_batches_train))
            input_features = blocks[0].srcdata['features']
            # recency (TO BE CLEANED)
            recency_scores = None
            if use_recency:
                recency_scores = pos_g.edata['recency']

            _, pos_score, neg_score = model(blocks,
                                            input_features,
                                            pos_g,
                                            neg_g,
                                            embedding_layer,
                                            )
            loss = loss_fn(pos_score,
                           neg_score,
                           delta,
                           neg_sample_size,
                           use_recency=use_recency,
                           recency_scores=recency_scores,
                           remove_false_negative=remove_false_negative,
                           negative_mask=negative_mask,
                           cuda=cuda,
                           device=device,
                           )

            if epoch > 0:  # For the epoch 0, no training (just report loss)
                loss.backward()
                opt.step()
            total_loss += loss.item()

            if epoch == 0 and i > 10:
                break  # For the epoch 0, report loss on only subset

        train_avg_loss = total_loss / i
        model.train_loss_list.append(train_avg_loss)

        print('VALIDATION LOSS')
        model.eval()
        with torch.no_grad():
            total_loss = 0
            i = 0
            for _, pos_g, neg_g, blocks in edgeloader_valid:
                i += 1
                if i % 10 == 0:
                    print("Edge batch {} out of {}".format(i, num_batches_val_loss))

                # Negative mask
                negative_mask = {}
                if remove_false_negative:
                    nids = neg_g.ndata[dgl.NID]
                    for etype in pos_g.canonical_etypes:
                        neg_src, neg_dst = neg_g.edges(etype=etype)
                        neg_src = nids[etype[0]][neg_src]
                        neg_dst = nids[etype[2]][neg_dst]
                        negative_mask_tensor = valid_graph.has_edges_between(neg_src, neg_dst, etype=etype)
                        negative_mask[etype] = negative_mask_tensor.type(torch.float)
                        if cuda:
                            negative_mask[etype] = negative_mask[etype].to(device)

                if cuda:
                    blocks = [b.to(device) for b in blocks]
                    pos_g = pos_g.to(device)
                    neg_g = neg_g.to(device)

                input_features = blocks[0].srcdata['features']
                _, pos_score, neg_score = model(blocks,
                                                input_features,
                                                pos_g,
                                                neg_g,
                                                embedding_layer,
                                                )
                # recency (TO BE CLEANED)
                recency_scores = None
                if use_recency:
                    recency_scores = pos_g.edata['recency']

                val_loss = loss_fn(pos_score,
                                   neg_score,
                                   delta,
                                   neg_sample_size,
                                   use_recency=use_recency,
                                   recency_scores=recency_scores,
                                   remove_false_negative=remove_false_negative,
                                   negative_mask=negative_mask,
                                   cuda=cuda,
                                   device=device,
                                   )
                total_loss += val_loss.item()
                # print(val_loss.item())
            val_avg_loss = total_loss / i
            model.val_loss_list.append(val_avg_loss)

        ############
        # METRICS PER EPOCH 
        if get_metrics and epoch % 10 == 1:
            model.eval()
            with torch.no_grad():
                # training metrics
                print('TRAINING METRICS')
                y = get_embeddings(train_graph,
                                   out_dim,
                                   model,
                                   nodeloader_subtrain,
                                   num_batches_subtrain,
                                   cuda,
                                   device,
                                   embedding_layer,
                                   )

                train_precision, train_recall, train_coverage = get_metrics_at_k(y,
                                                                                 train_graph,
                                                                                 model,
                                                                                 out_dim,
                                                                                 ground_truth_subtrain,
                                                                                 bought_eids,
                                                                                 k,
                                                                                 False,  # Remove already bought
                                                                                 cuda,
                                                                                 device,
                                                                                 pred,
                                                                                 use_popularity,
                                                                                 weight_popularity)

                # validation metrics
                print('VALIDATION METRICS')
                y = get_embeddings(valid_graph,
                                   out_dim,
                                   model,
                                   nodeloader_valid,
                                   num_batches_val_metrics,
                                   cuda,
                                   device,
                                   embedding_layer,
                                   )

                val_precision, val_recall, val_coverage = get_metrics_at_k(y,
                                                                           valid_graph,
                                                                           model,
                                                                           out_dim,
                                                                           ground_truth_valid,
                                                                           bought_eids,
                                                                           k,
                                                                           remove_already_bought,
                                                                           cuda,
                                                                           device,
                                                                           pred,
                                                                           use_popularity,
                                                                           weight_popularity
                                                                           )
                sentence = '''Epoch {:05d} || TRAINING Loss {:.5f} | Precision {:.3f}% | Recall {:.3f}% | Coverage {:.2f}% 
                || VALIDATION Loss {:.5f} | Precision {:.3f}% | Recall {:.3f}% | Coverage {:.2f}% '''.format(
                    epoch, train_avg_loss, train_precision * 100, train_recall * 100, train_coverage * 100,
                    val_avg_loss, val_precision * 100, val_recall * 100, val_coverage * 100)
                print(sentence)
                save_txt(sentence, result_filepath, mode='a')

                model.train_precision_list.append(train_precision * 100)
                model.train_recall_list.append(train_recall * 100)
                model.train_coverage_list.append(train_coverage * 10)
                model.val_precision_list.append(val_precision * 100)
                model.val_recall_list.append(val_recall * 100)
                model.val_coverage_list.append(val_coverage * 10)  # just *10 for viz purposes

                # Visualization of best metric
                if val_recall > max_metric:
                    max_metric = val_recall
                    best_metrics = {'recall': val_recall, 'precision': val_precision, 'coverage': val_coverage}

        else:
            sentence = "Epoch {:05d} | Training Loss {:.5f} | Validation Loss {:.5f} | ".format(
                epoch, train_avg_loss, val_avg_loss)
            print(sentence)
            save_txt(sentence, result_filepath, mode='a')

        if val_avg_loss < min_loss:
            min_loss = val_avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter == patience:
            break

        elapsed = time.time() - start_time
        result_to_save = f'Epoch took {timedelta(seconds=elapsed)} \n'
        print(result_to_save)
        save_txt(result_to_save, result_filepath, mode='a')

    viz = {'train_loss_list': model.train_loss_list,
           'train_precision_list': model.train_precision_list,
           'train_recall_list': model.train_recall_list,
           'train_coverage_list': model.train_coverage_list,
           'val_loss_list': model.val_loss_list,
           'val_precision_list': model.val_precision_list,
           'val_recall_list': model.val_recall_list,
           'val_coverage_list': model.val_coverage_list}

    print('Training completed.')
    return model, viz, best_metrics  # model will already be to 'cuda' device?


def get_embeddings(g,
                   out_dim: int,
                   trained_model,
                   nodeloader_test,
                   num_batches_valid: int,
                   cuda: bool = False,
                   device=None,
                   embedding_layer: bool = True):
    """
    Fetch the embeddings for all the nodes in the nodeloader.

    Nodeloader is preferable when computing embeddings because we can specify which nodes to compute the embedding for,
    and only have relevant nodes in the computational blocks. Whereas Edgeloader is preferable for training, because
    we generate negative edges also.
    """
    if cuda:  # model is already on device?
        trained_model = trained_model.to(device)
    i2 = 0
    y = {ntype: torch.zeros(g.num_nodes(ntype), out_dim)
         for ntype in g.ntypes}
    if cuda:  # not sure if I need to put the 'result' tensor to device
        y = {ntype: torch.zeros(g.num_nodes(ntype), out_dim).to(device)
             for ntype in g.ntypes}
    for input_nodes, output_nodes, blocks in nodeloader_test:
        i2 += 1
        if i2 % 10 == 0:
            print("Computing embeddings: Batch {} out of {}".format(i2, num_batches_valid))
        if cuda:
            blocks = [b.to(device) for b in blocks]
        input_features = blocks[0].srcdata['features']
        if embedding_layer:
            input_features['user'] = trained_model.user_embed(input_features['user'])
            input_features['item'] = trained_model.item_embed(input_features['item'])
            if 'sport' in input_features.keys():
                input_features['sport'] = trained_model.sport_embed(input_features['sport'])
        h = trained_model.get_repr(blocks, input_features)
        for ntype in h.keys():
            y[ntype][output_nodes[ntype]] = h[ntype]
    return y
