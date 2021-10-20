from utils import compute_ppr, compute_heat, preprocess_features, normalize_adj
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
import numpy as np
import os
import sys
from torch_geometric.datasets import Planetoid, TUDataset
import torch_geometric.transforms as T
import torch
import os.path as osp


def load(dataset):
    if dataset == "acm":
        x_path = osp.join("data", dataset + r".txt")
        label_path = osp.join("data", dataset + r"_label.txt")
        adj_path = osp.join("data", dataset + r"_graph.txt")
        x = np.loadtxt(x_path, dtype=float)
        label = np.loadtxt(label_path, dtype=int)
        n, _ = x.shape

        idx = np.array([i for i in range(n)], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(adj_path, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        coo_adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(n, n), dtype=np.float32)
        row, col = coo_adj.row, coo_adj.col
        adj = np.zeros((n, n), dtype=np.float)
        adj[row, col] = 1.0

        A = adj
        d = np.sum(A, axis=1).astype(float)
        adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
        diff = compute_ppr(torch.from_numpy(A), alpha=0.15).numpy()
        # diff = compute_heat(torch.from_numpy(A), t=6).numpy()

        return adj, diff, x, label, None, None, None, A, d
    else:
        ds = Planetoid(root="./data", name=dataset, transform=T.ToSparseTensor())
        data = ds[0]
        adj = data.adj_t.to_dense().float()
        # diff = compute_ppr(adj, alpha=0.08).numpy()
        diff = compute_heat(adj, t=11).numpy()
        adj = adj.numpy()
        # node_features = data.x.numpy()
        node_features = (data.x / (data.x.sum(axis = 1, keepdims=True) + sys.float_info.epsilon)).numpy()
        # node_features = (data.x / (torch.norm(data.x, p=2, dim=1, keepdim=True) + sys.float_info.epsilon)).numpy()
        labels = data.y.numpy()

        idx_train = torch.nonzero(data.train_mask, as_tuple=False).squeeze().numpy()
        idx_val = torch.nonzero(data.val_mask, as_tuple=False).squeeze().numpy()
        idx_test = torch.nonzero(data.test_mask, as_tuple=False).squeeze().numpy()

        if dataset == 'citeseer':
            node_features = preprocess_features(node_features)
            epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
            avg_degree = np.sum(adj) / adj.shape[0]
            epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
                                          for e in epsilons])]
            diff[diff < epsilon] = 0.0
            scaler = MinMaxScaler()
            scaler.fit(diff)
            diff = scaler.transform(diff)
        A = adj
        d = np.sum(A, axis=1).astype(float)
        adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
    
        return adj, diff, node_features, labels, idx_train, idx_val, idx_test, A, d



if __name__ == '__main__':
    load('citeseer')
    print("OK!")
