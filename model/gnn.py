import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GCN2Conv, GATConv, SAGEConv, GINConv, ChebConv, SGConv

num_classes = {
    'twitch': 2,
    'facebook100': 2,
    'cora': 7,
    'citeseer': 6,
    'politifact': 2,
    'gossipcop': 2
}


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, in_sparse_feat_channels=None,
                 cross_graph_label_rel_batch_size=128, conv='sage'
                 ):
        super(SAGE, self).__init__()
        conv_op = SAGEConv if conv == 'sage' else GATConv if conv == 'GAT' else GCN2Conv if conv == 'gcn' \
            else GINConv if conv == 'gin' else ChebConv
        self.model_type = conv
        self.spare_feature_encoder = torch.nn.Linear(in_sparse_feat_channels, in_channels) if in_sparse_feat_channels \
                                                                                              is not None else None
        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_op(in_feats=in_channels, out_feats=hidden_channels, aggregator_type='mean'))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                conv_op(hidden_channels, hidden_channels, aggregator_type='mean'))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(conv_op(hidden_channels, out_channels, aggregator_type='mean'))

        self.dropout = dropout
        self.cross_graph_label_rel_batch_size = cross_graph_label_rel_batch_size
        self.feature_IB_fc = torch.nn.Linear(in_features=in_channels, out_features=hidden_channels, bias=False)

    def reset_parameters(self):
        self.spare_feature_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.feature_IB_fc.reset_parameters()

    # compute relational ground truth for S-GIB
    def compute_label_rel_IB_loss(self, nodes_feature, labels):
        select_node_batch = torch.cat(nodes_feature, dim=0)
        select_label_batch = torch.cat(labels, dim=0)
        label_same = select_label_batch.unsqueeze(dim=1) == select_label_batch.unsqueeze(dim=0)
        label_diff = select_label_batch.unsqueeze(dim=1) != select_label_batch.unsqueeze(dim=0)
        x_after = select_node_batch
        x_rel = torch.mm(x_after, select_node_batch.T)
        sig_x_rel = torch.sigmoid(x_rel)

        same_label_eles = torch.log(sig_x_rel)[label_same]
        diff_label_eles = torch.log(1 - sig_x_rel + 1e-20)[label_diff]
        same_label_pairs = -torch.sum(same_label_eles)
        diff_label_pairs = -torch.sum(diff_label_eles)
        return same_label_pairs, diff_label_pairs

    # compute I-GIB loss
    def compute_first_last_feature_IB_loss(self, first_layer_feature, last_layer_feature):
        first_features = torch.cat(first_layer_feature, dim=0)
        last_features = torch.cat(last_layer_feature,  dim=0)
        first_features_after = self.feature_IB_fc(first_features)
        sig_features_rel = torch.sigmoid(torch.sum(torch.mul(first_features_after, last_features), dim=-1))
        return -torch.sum(torch.log(1 - sig_features_rel + 1e-20))

    # compute S-GIB loss
    def compute_first_last_feature_IB_loss_2nd_order(self, first_layer_feature, last_layer_feature):
        first_features = torch.cat(first_layer_feature, dim=0)
        last_features = torch.cat(last_layer_feature, dim=0)
        first_features_rel = torch.mm(first_features, first_features.T)
        last_feature_rel = torch.mm(last_features, last_features.T)
        f_l_rel = first_features_rel - last_feature_rel
        return torch.norm(f_l_rel, p='fro')


    def forward(self, graph, inputs, **kwargs):
        x = self.spare_feature_encoder(inputs) if self.spare_feature_encoder is not None else inputs
        first_l_x = x
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(graph, x) if self.model_type != 'gcn' else conv(graph, x, first_l_x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        feat = x
        x = self.convs[-1](graph, x) if self.model_type != 'gcn' else self.convs[-1](graph, x, first_l_x)
        return x.log_softmax(dim=-1), first_l_x, feat


class PoolSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, in_sparse_feat_channels=None,
                 cross_graph_label_rel_batch_size=128
                 ):
        super(PoolSAGE, self).__init__()

        self.spare_feature_encoder = torch.nn.Linear(in_sparse_feat_channels, in_channels) if in_sparse_feat_channels \
                                                                                              is not None else None
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_feats=in_channels, out_feats=hidden_channels, aggregator_type='mean'))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggregator_type='mean'))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggregator_type='mean'))

        self.dropout = dropout
        self.cross_graph_label_rel_batch_size = cross_graph_label_rel_batch_size
        self.feature_IB_fc = torch.nn.Linear(in_features=in_channels, out_features=hidden_channels, bias=False)

    def reset_parameters(self):
        self.spare_feature_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.feature_IB_fc.reset_parameters()

    # compute relational ground truth for S-GIB
    def compute_label_rel_IB_loss(self, nodes_feature, labels):
        select_node_batch = torch.cat(nodes_feature, dim=0)
        select_label_batch = torch.cat(labels, dim=0)
        label_same = select_label_batch.unsqueeze(dim=1) == select_label_batch.unsqueeze(dim=0)
        label_diff = select_label_batch.unsqueeze(dim=1) != select_label_batch.unsqueeze(dim=0)
        x_after = select_node_batch
        x_rel = torch.mm(x_after, select_node_batch.T)
        sig_x_rel = torch.sigmoid(x_rel)
        same_label_pairs = -torch.sum(torch.log(sig_x_rel)[label_same])

        diff_label_pairs = -torch.sum(torch.log(1 - sig_x_rel + 1e-20)[label_diff])
        return same_label_pairs, diff_label_pairs

    # compute I-GIB loss
    def compute_first_last_feature_IB_loss(self, first_layer_feature, last_layer_feature):
        first_features = torch.cat(first_layer_feature, dim=0)
        last_features = torch.cat(last_layer_feature,  dim=0)
        first_features_after = self.feature_IB_fc(first_features)
        sig_features_rel = torch.sigmoid(torch.sum(torch.mul(first_features_after, last_features), dim=-1))
        return -torch.sum(torch.log(1 - sig_features_rel + 1e-20))

    # compute S-GIB loss
    def compute_first_last_feature_IB_loss_2nd_order(self, first_layer_feature, last_layer_feature):
        first_features = torch.cat(first_layer_feature, dim=0)
        last_features = torch.cat(last_layer_feature, dim=0)
        first_features_rel = torch.mm(first_features, first_features.T)
        last_feature_rel = torch.mm(last_features, last_features.T)
        return torch.norm(first_features_rel - last_feature_rel, p='fro')


    def forward(self, graph, inputs, **kwargs):
        x = self.spare_feature_encoder(inputs) if self.spare_feature_encoder is not None else inputs
        first_l_x = x
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(graph, x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        feat = x
        x = self.convs[-1](graph, x)
        x = torch.mean(x, dim=0, keepdim=True)
        return x.log_softmax(dim=-1), first_l_x, feat


def build_model(args, device, in_sparse_feature_dim):
    if args.dataset in ['politifact', 'gossipcop']:
        # graph level
        assert args.model == 'sage'
        return PoolSAGE(
            in_channels=args.in_channels, hidden_channels=args.hidden_channels,
            out_channels=num_classes[args.dataset], num_layers=args.num_layers,
            dropout=args.dropout, in_sparse_feat_channels=in_sparse_feature_dim,
            cross_graph_label_rel_batch_size=args.cross_graph_label_rel_batch_size
        ).to(device)

    # node level
    return SAGE(in_channels=args.in_channels, hidden_channels=args.hidden_channels,
                out_channels=num_classes[args.dataset], num_layers=args.num_layers,
                dropout=args.dropout, in_sparse_feat_channels=in_sparse_feature_dim,
                cross_graph_label_rel_batch_size=args.cross_graph_label_rel_batch_size, conv=args.model
                ).to(device)
