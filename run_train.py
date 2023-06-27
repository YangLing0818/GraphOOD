from load_data import *
import numpy as np
from model.gnn import *
from sklearn.metrics import roc_auc_score, accuracy_score
import argparse
import os

import random
import pickle
from datetime import datetime
import torch
from logger import Logger
from utils import *

sparse_feature_dict = {
    'twitch': 3170,
    'facebook100': 6,
    'cora': 1433,
    'citeseer': 3703,
    'politifact': 768,
    'gossipcop': 768
}

loss_name_weight_dict = {
    'risk_extrapolation_loss': 'risk_var_weight',
    'cross_graph_label_rel_IB_loss_same': 'label_rel_IB_loss_weight_1',
    'cross_graph_label_rel_IB_loss_diff': 'label_rel_IB_loss_weight_2',
    'first_last_1nd_loss': 'in_out_feature_instance_loss_weight',
    'first_last_2nd_loss': 'in_out_feature_rel_loss_weight'
}


def parse_args():
    parser = argparse.ArgumentParser(description='IS-GIB')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--risk_extrapolation_loss', action='store_true')

    parser.add_argument('--cross_graph_label_rel_IB_loss', action='store_true')
    parser.add_argument('--first_last_layer_IB_loss', action='store_true')
    parser.add_argument('--first_last_layer_2nd_IB_loss', action='store_true')

    parser.add_argument('--cross_graph_label_rel_batch_size', type=int, default=128)
    parser.add_argument('--risk_var_weight', type=float, default=1)
    parser.add_argument('--metric', type=str, default='roc_auc')
    parser.add_argument('--label_rel_IB_loss_weight_1', type=float, default=1e-2)
    parser.add_argument('--label_rel_IB_loss_weight_2', type=float, default=1e-5)
    parser.add_argument('--in_out_feature_rel_loss_weight', type=float, default=1e-11)
    parser.add_argument('--in_out_feature_instance_loss_weight', type=float, default=1)
    parser.add_argument('--model', type=str, default='sage')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--train_graph_list', type=str, default="['DE', 'ES', 'FR']")
    parser.add_argument('--val_graph_list', type=str, default="['ENGB']")
    parser.add_argument('--test_graph_list', type=str, default=None)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--in_channels', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch_runs_base_dir', type=str, default=None)
    parser.add_argument('--raw_pickle_dir', type=str, default='raw_results/')
    parser.add_argument('--pic_dir', type=str, default='pic_results/')
    parser.add_argument('--append_best_file', type=str, default=None)
    parser.add_argument('--feature_noise', action='store_true')
    parser.add_argument('--noise_mu_lwb', type=float, default=-0.05)
    parser.add_argument('--noise_mu_upb', type=float, default=0.05)
    parser.add_argument('--noise_sigma_lwb', type=float, default=1.)
    parser.add_argument('--noise_sigma_upb', type=float, default=2.)
    parser.add_argument('--save_model_dir', type=str, default='./saved_models/')

    args = parser.parse_args()
    if args.batch_runs_base_dir is not None:
        args.raw_pickle_dir = os.path.join(args.batch_runs_base_dir, args.raw_pickle_dir)
        args.pic_dir = os.path.join(args.batch_runs_base_dir, args.pic_dir)
        if args.append_best_file is not None:
            args.append_best_file = os.path.join(args.batch_runs_base_dir, args.append_best_file)
    args.train_graph_list = eval(args.train_graph_list)
    args.val_graph_list = eval(args.val_graph_list)
    args.test_graph_list = eval(args.test_graph_list) if args.test_graph_list is not None else None
    print(args)
    return args


def train(model, graph_list, optimizer, args):
    model.train()
    optimizer.zero_grad()
    loss_list = []
    feat_list = []
    select_first_layer_feat_batch = []
    select_node_batch = []
    select_label_batch = []
    loss_dict = {}
    for i, graph in enumerate(graph_list):
        out, first_layer_feat, last_layer_feat = model(graph, graph.ndata['feat'])
        feat_list.append(last_layer_feat)
        loss = F.nll_loss(out, graph.ndata['label'])
        loss_list.append(loss.unsqueeze(0))
        if args.cross_graph_label_rel_IB_loss or args.first_last_layer_IB_loss or args.first_last_layer_2nd_IB_loss:
            idx = list(range(graph.ndata['label'].shape[0]))
            random.shuffle(idx)
            rand_idx = idx[:args.cross_graph_label_rel_batch_size]
            select_first_layer_feat_batch.append(first_layer_feat[rand_idx])
            select_node_batch.append(last_layer_feat[rand_idx])
            select_label_batch.append(graph.ndata['label'][rand_idx])

    loss_list = torch.cat(loss_list, dim=0)
    loss = torch.mean(loss_list)
    loss_dict['erm'] = loss.item()
    if args.risk_extrapolation_loss and len(graph_list) > 1:
        risk_ext_loss = torch.var(loss_list, dim=0)
        loss_dict['risk_extrapolation_loss'] = risk_ext_loss.item()
        loss += args.risk_var_weight * risk_ext_loss

    if args.cross_graph_label_rel_IB_loss:
        label_rel_ib_loss_same_pair, label_rel_ib_loss_diff_pair = \
            model.compute_label_rel_IB_loss(select_node_batch, select_label_batch)
        loss_dict['cross_graph_label_rel_IB_loss_same'] = label_rel_ib_loss_same_pair.item()
        loss_dict['cross_graph_label_rel_IB_loss_diff'] = label_rel_ib_loss_diff_pair.item()
        loss += args.label_rel_IB_loss_weight_1 * label_rel_ib_loss_same_pair + \
            args.label_rel_IB_loss_weight_2 * label_rel_ib_loss_diff_pair

    if args.first_last_layer_IB_loss:
        first_last_1nd_loss = model.compute_first_last_feature_IB_loss(select_first_layer_feat_batch, select_node_batch)
        loss_dict['first_last_1nd_loss'] = first_last_1nd_loss.item()
        loss += first_last_1nd_loss * args.in_out_feature_instance_loss_weight

    if args.first_last_layer_2nd_IB_loss:
        first_last_2nd_loss = model.compute_first_last_feature_IB_loss_2nd_order(select_first_layer_feat_batch, select_node_batch)
        loss_dict['first_last_2nd_loss'] = first_last_2nd_loss.item()
        loss += first_last_2nd_loss * args.in_out_feature_rel_loss_weight

    loss.backward()
    loss_dict['final_loss'] = loss.item()
    optimizer.step()
    # return loss.item()
    return loss_dict


def train_graph_level(model, graph_list, optimizer, args):
    model.train()
    optimizer.zero_grad()
    loss_list = []
    first_feat_list = []
    last_feat_list = []
    select_first_layer_feat_batch = []
    select_node_batch = []
    select_label_batch = []
    loss_dict = {}
    for i, graph in enumerate(graph_list):
        out, first_layer_feat, last_layer_feat = model(graph, graph.ndata['feat'])
        first_feat_list.append(first_layer_feat)
        last_feat_list.append(last_layer_feat)
        select_node_batch.append(torch.mean(last_layer_feat, dim=0, keepdim=True))
        select_label_batch.append(graph.ndata['label'][0].unsqueeze(0))
        select_first_layer_feat_batch.append(torch.mean(first_layer_feat, dim=0, keepdim=True))
        loss = F.nll_loss(out, graph.ndata['label'][0].unsqueeze(0))
        loss_list.append(loss.unsqueeze(0))

    loss_list = torch.cat(loss_list, dim=0)
    loss = torch.mean(loss_list)
    loss_dict['erm'] = loss.item()
    if args.risk_extrapolation_loss and len(graph_list) > 1:
        risk_ext_loss = torch.var(loss_list, dim=0)
        loss_dict['risk_extrapolation_loss'] = risk_ext_loss.item()
        loss += args.risk_var_weight * risk_ext_loss

    if args.cross_graph_label_rel_IB_loss:
        label_rel_ib_loss_same_pair, label_rel_ib_loss_diff_pair = \
            model.compute_label_rel_IB_loss(select_node_batch, select_label_batch)
        loss_dict['cross_graph_label_rel_IB_loss_same'] = label_rel_ib_loss_same_pair.item()
        loss_dict['cross_graph_label_rel_IB_loss_diff'] = label_rel_ib_loss_diff_pair.item()
        loss += args.label_rel_IB_loss_weight_1 * label_rel_ib_loss_same_pair + \
                args.label_rel_IB_loss_weight_2 * label_rel_ib_loss_diff_pair

    if args.first_last_layer_IB_loss:
        first_last_1nd_loss = model.compute_first_last_feature_IB_loss(select_first_layer_feat_batch,
                                                                       select_node_batch)
        loss_dict['first_last_1nd_loss'] = first_last_1nd_loss.item()
        loss += first_last_1nd_loss * args.in_out_feature_instance_loss_weight

    if args.first_last_layer_2nd_IB_loss:
        first_last_2nd_loss = model.compute_first_last_feature_IB_loss_2nd_order(select_first_layer_feat_batch,
                                                                                 select_node_batch)
        loss_dict['first_last_2nd_loss'] = first_last_2nd_loss.item()
        loss += first_last_2nd_loss * args.in_out_feature_rel_loss_weight

    loss.backward()
    loss_dict['final_loss'] = loss.item()
    optimizer.step()
    # return loss.item()
    return loss_dict


@torch.no_grad()
def test(model, train_graph_list, val_graph_list, test_graph_list, args):
    model.eval()

    def loop_eval(graph_list):
        return_list = []
        return_first_feat_list = []
        return_feat_list = []
        for graph in graph_list:
            out, first_l_feat, feat = model(graph, graph.ndata['feat'])
            return_feat_list.append(feat.cpu())
            return_first_feat_list.append(first_l_feat.cpu())
            y_pred = out.argmax(dim=-1, keepdim=True).cpu()  # (num_nodes, 1)
            y_score = out[:, 1].cpu()

            y_true = graph.ndata['label'].cpu()

            y_pred = y_pred.numpy()[y_true != -100]
            y_score = y_score.numpy()[y_true != -100]
            y_true = y_true.numpy()[y_true != -100]

            if args.metric == 'roc_auc':
                assert args.dataset == 'twitch'
                return_list.append(roc_auc_score(y_true=y_true,
                                                 y_score=y_score))
            else:
                return_list.append(accuracy_score(y_true=y_true,
                                                  y_pred=y_pred))
        return return_list, return_feat_list, return_first_feat_list

    train_metrics, train_feat_list, train_first_feat_list = loop_eval(train_graph_list)
    val_metrics, val_feat_list, val_first_feat_list = loop_eval(val_graph_list)
    test_metrics, test_feat_list, test_first_feat_list = loop_eval(test_graph_list)

    return np.mean(train_metrics), np.mean(val_metrics), np.mean(test_metrics), \
           train_feat_list, val_feat_list, test_feat_list, train_first_feat_list, \
           val_first_feat_list, test_first_feat_list


@torch.no_grad()
def test_graph_level(model, train_graph_list, val_graph_list, test_graph_list, args):
    model.eval()

    def loop_eval(graph_list):
        return_feat_list = []
        y_pred_list = []
        y_true_list = []
        for graph in graph_list:
            out, first_l_feat, feat = model(graph, graph.ndata['feat'])
            return_feat_list.append(feat.cpu())
            y_pred_list.append(out.argmax(dim=-1).cpu().item())  # (num_nodes, 1)
            y_true_list.append(graph.ndata['label'][0].cpu().item())

        metric = accuracy_score(y_pred=y_pred_list, y_true=y_true_list)

        return metric, return_feat_list

    train_metrics, train_feat_list = loop_eval(train_graph_list)
    val_metrics, val_feat_list = loop_eval(val_graph_list)
    test_metrics, test_feat_list = loop_eval(test_graph_list)

    return train_metrics, val_metrics, test_metrics, \
        train_feat_list, val_feat_list, test_feat_list


def main():
    args = parse_args()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.seed is not None:
        set_seed(args.seed)
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    if not os.path.exists(args.raw_pickle_dir):
        os.makedirs(args.raw_pickle_dir)

    train_graph_list, val_graph_list, test_graph_list = \
        load(dataset_name=args.dataset, train_graph_list=args.train_graph_list,
             val_graph_list=args.val_graph_list, test_graph_list=args.test_graph_list, device=device,
             add_noise_feature=args.feature_noise,
             mu_lwb=args.noise_mu_lwb,
             mu_upb=args.noise_mu_upb,
             sigma_lwb=args.noise_sigma_lwb,
             sigma_upb=args.noise_sigma_upb)
    model = build_model(args, device, sparse_feature_dict[args.dataset])
    logger = Logger(args.runs, args)

    test_labels = [graph.ndata['label'].cpu() for graph in test_graph_list]
    with open(args.raw_pickle_dir + 'test_labels.pkl', 'wb') as f:
        pickle.dump(test_labels, f)

    train_labels = [graph.ndata['label'].cpu() for graph in train_graph_list]
    with open(args.raw_pickle_dir + 'train_labels.pkl', 'wb') as f:
        pickle.dump(train_labels, f)

    for run in range(args.runs):
        model.reset_parameters()
        np.random.seed(run)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_func = train if args.dataset not in ['politifact', 'gossipcop'] else train_graph_level
        test_func = test if args.dataset not in ['politifact', 'gossipcop'] else test_graph_level

        for epoch in range(1, 1 + args.epochs):
            loss_dict = train_func(model, train_graph_list, optimizer, args)
            train_r, dev_r, test_r, train_feat, dev_feat, test_feat, train_first_feat, dev_first_feat, test_first_feat = \
                test_func(model, train_graph_list, val_graph_list, test_graph_list, args)
            result = [train_r, dev_r, test_r]
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_rocauc, valid_rocauc, test_rocauc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      # f'Loss: {loss:.4f}, '
                      f'Train: {train_rocauc:.4f}, '
                      f'Valid: {valid_rocauc:.4f}, '
                      f'Test: {test_rocauc:.4f}.')
                for k, v in loss_dict.items():
                    if k == 'erm' or k == 'final_loss':
                        print(f'loss: {k}, value {v}')
                    else:
                        print(f'loss: {k}, value {v}, weight {getattr(args, loss_name_weight_dict[k])} ')

        logger.print_statistics(run)
    results = logger.print_statistics()

    model_config = args.model
    loss_config = 'risk_loss' if args.risk_extrapolation_loss \
        else 'normal_loss'
    now_time = str(datetime.now())

    if args.batch_runs_base_dir is not None:
        config_file = os.path.join(args.batch_runs_base_dir, 'args.txt')
        if not os.path.exists(config_file):
            with open(config_file, 'w') as f:
                f.write(str(args))

    if args.append_best_file is not None:
        with open(args.append_best_file, 'a+') as f:
            best_test = str(np.max(results[3].numpy())) + '\n'
            f.write(best_test)
    with open(args.raw_pickle_dir + 'raw_{}.pkl'.format(now_time), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
