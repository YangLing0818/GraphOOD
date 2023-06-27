import scipy.io
import numpy as np
import scipy.sparse
import torch
import csv
import json
import dgl
import os
from os import path
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.fakenews import FakeNewsDataset
from sklearn.preprocessing import label_binarize
import random


def load_twitch_raw(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = f"dataset/twitch/{lang}"
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2] == "True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]

    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)),
                                 (np.array(src), np.array(targ))),
                                shape=(n, n))
    features = np.zeros((n, 3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    # features = features[:, np.sum(features, axis=0) != 0]  # remove zero cols
    new_label = label[reorder_node_ids]
    label = new_label

    return A, label, features


def load_twitch_dgl(train_lang_list, val_lang_list, device, add_noise_feature=False,
                    mu_lwb=-0.05, mu_upb=0.05, sigma_lwb=1., sigma_upb=2.):
    assert not add_noise_feature
    train_graph_list = []
    val_graph_list = []
    test_graph_list = []
    for lang in ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW']:
        A, label, features = load_twitch_raw(lang)
        dgl_g = dgl.from_scipy(A)
        if add_noise_feature:
            sigma = np.random.uniform(sigma_lwb, sigma_upb)
            mu = np.random.uniform(mu_lwb, mu_upb)
            # features = features + np.random.randn(features.shape[0], features.shape[1])  # 这个实验效果好
            features = features + np.random.normal(mu, sigma, features.shape)
        dgl_g.ndata['feat'] = torch.tensor(features, dtype=torch.float32)
        dgl_g.ndata['label'] = torch.tensor(label)
        if lang in train_lang_list:
            train_graph_list.append(dgl_g.to(device))
        elif lang in val_lang_list:
            val_graph_list.append(dgl_g.to(device))
        else:
            test_graph_list.append(dgl_g.to(device))
    return train_graph_list, val_graph_list, test_graph_list


def load(dataset_name, train_graph_list, val_graph_list, test_graph_list, device, add_noise_feature=False,
         mu_lwb=-0.05, mu_upb=0.05, sigma_lwb=1., sigma_upb=2.):
    if dataset_name == 'twitch':
        return load_twitch_dgl(train_lang_list=train_graph_list, val_lang_list=val_graph_list, device=device,
                               add_noise_feature=add_noise_feature, mu_lwb=mu_lwb, mu_upb=mu_upb, sigma_lwb=sigma_lwb,
                               sigma_upb=sigma_upb)


if __name__ == '__main__':
    pass
