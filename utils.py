import numpy as np
import networkx as nx
import torch
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp


def compute_ppr(A, alpha=0.2, self_loop=True):
    I = torch.eye(A.shape[0], dtype=torch.float).to(A.device)
    if self_loop:
        A = A + I
    d = torch.sum(A, dim=1)
    d_inv = torch.pow(d, -0.5)
    d_inv[d_inv == np.inf] = 0
    D_inv = torch.diag(d_inv)
    AT = torch.matmul(D_inv, torch.matmul(A, D_inv))
    ppr = alpha * torch.inverse(I - (1 - alpha) * AT)
    return ppr

def compute_heat(A, t=9, self_loop=True):
    I = torch.eye(A.shape[0], dtype=torch.float).to(A.device)
    if self_loop:
        A = A + I
    d = torch.sum(A, dim=1)
    d_inv = torch.pow(d, -0.5)
    d_inv[d_inv == np.inf] = 0
    D_inv = torch.diag(d_inv)
    AT = torch.matmul(D_inv, torch.matmul(A, D_inv))
    heat = torch.exp(t * AT) / torch.exp(torch.tensor(t, dtype=torch.float))
    return heat

def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj, self_loop=True):
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def soft(score, lambd):
    score = torch.exp(score)
    score = torch.pow(score, lambd)
    sum = torch.sum(score, dim=1, keepdim=True)
    norm_score = score / sum
    return norm_score

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
