import numpy as np
import torch
import torch.nn as nn
import argparse
from dataset import load
import os.path as osp
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier


def get_args():
    parser = argparse.ArgumentParser(description='Args for FSC')
    parser.add_argument('-num_epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('-patience', type=int, default=20, help='patience')
    parser.add_argument('-learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-l2_coef', type=float, default=0.0, help='l2 coefficient')
    parser.add_argument('-hid_units', type=int, default=512, help='number of hidden units')
    parser.add_argument('-dataset', type=str, default="citeseer", help='dataset')
    args, _ = parser.parse_known_args()
    return args


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, attr, adj):
        repr = self.fc(attr)

        out = torch.bmm(adj, repr)

        if self.bias is not None:
            out += self.bias
        return self.act(out)


class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.Tnet = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, a1, a2, h1, h2, h3, h4):

        a_h1 = torch.unsqueeze(a1, 1)
        a_h1 = a_h1.expand_as(h1).contiguous()
        a_h2 = torch.unsqueeze(a2, 1)
        a_h2 = a_h2.expand_as(h2).contiguous()

        # positive pair
        pp_1 = torch.squeeze(self.Tnet(h2, a_h1), 2)
        pp_2 = torch.squeeze(self.Tnet(h1, a_h2), 2)

        # negetive pair
        np_1 = torch.squeeze(self.Tnet(h4, a_h1), 2)
        np_2 = torch.squeeze(self.Tnet(h3, a_h2), 2)

        logits = torch.cat((pp_1, pp_2, np_1, np_2), 1)

        return logits


class Model(nn.Module):
    def __init__(self, n_fa_in, n_sa_in, n_h):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_fa_in, n_h)
        self.gcn2 = GCN(n_sa_in, n_h)
        self.read = Readout()

        self.sigmoid = nn.Sigmoid()

        self.discrim = Discriminator(n_h)

    def forward(self, fa, shuf_fa, adj, sa, shuf_sa, msk):
        h_1 = self.gcn1(fa, adj)
        a_1 = self.read(h_1, msk)
        a_1 = self.sigmoid(a_1)

        h_2 = self.gcn2(sa, adj)
        a_2 = self.read(h_2, msk)
        a_2 = self.sigmoid(a_2)

        h_3 = self.gcn1(shuf_fa, adj)
        h_4 = self.gcn2(shuf_sa, adj)

        ret = self.discrim(a_1, a_2, h_1, h_2, h_3, h_4)

        return ret, h_1, h_2

    def embed(self, fs, adj, msk):
        h_1 = self.gcn1(fs, adj)
        a = self.read(h_1, msk)

        return (h_1).detach(), a.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, repr):
        cls = torch.log_softmax(self.fc(repr), dim=-1)
        return cls


def train(dataset, args):
    nb_epochs = args.num_epochs
    patience = args.patience
    lr = args.learning_rate
    l2_coef = args.l2_coef
    hid_units = args.hid_units

    adj, diff, features, labels, idx_train, idx_val, idx_test, A, d = load(dataset)

    # normalize A
    I = np.eye(A.shape[0], dtype=np.float)
    A = A + I
    d_p = np.sum(A, axis=1)     # with self-loop
    d_p = d_p.reshape(-1, 1)
    dp_inv = 1 / d_p
    dp_inv[dp_inv == np.inf] = 0.0
    norm_A = dp_inv * A

    # utilize degree
    max_d = np.max(d).astype(int)
    D_onehot = np.zeros((d.shape[0], max_d+1), dtype=np.float)
    D_onehot[np.arange(d.shape[0]), d.astype(int)] = 1.0
    A_aug = np.linalg.matrix_power(A, 1).astype(bool).astype(float)
    D_aug = np.matmul(A_aug, D_onehot)
    row_sum = np.sum(D_aug, axis=1).astype(float)
    row_sum_inv = (1.0 / row_sum).reshape(-1, 1)
    D_aug = row_sum_inv * D_aug
    degree_size = D_aug.shape[1]

    ft_size = features.shape[1]
    nb_nodes = features.shape[0]
    nb_classes = np.unique(labels).shape[0]
    sample_size = nb_nodes
    batch_size = 4

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    lbl_p = torch.ones(batch_size, sample_size * 2)
    lbl_n = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_p, lbl_n), 1)


    model = Model(ft_size, degree_size, hid_units)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        lbl = lbl.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    losses = []
    for epoch in range(nb_epochs):
        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)


        # use the degree of nodes
        fa, sa, badj = [], [], []
        for i in idx:
            fa.append(features[i: i + sample_size])
            sa.append(D_aug[i: i + sample_size])
            badj.append(adj[i: i + sample_size, i: i + sample_size])

        fa = np.array(fa).reshape(batch_size, sample_size, ft_size)
        sa = np.array(sa).reshape(batch_size, sample_size, degree_size)
        badj = np.array(badj).reshape(batch_size, sample_size, sample_size)

        badj = torch.FloatTensor(badj)
        fa = torch.FloatTensor(fa)
        sa = torch.FloatTensor(sa)

        idx1 = np.random.permutation(sample_size)
        idx2 = np.random.permutation(sample_size)

        shuf_fa = fa[:, idx1, :]
        shuf_sa = sa[:, idx2, :]


        if torch.cuda.is_available():
            fa = fa.cuda()
            sa = sa.cuda()
            badj = badj.cuda()
            shuf_fa = shuf_fa.cuda()
            shuf_sa = shuf_sa.cuda()

        model.train()
        optimiser.zero_grad()

        logits, h1, h2 = model(fa, shuf_fa, badj, sa, shuf_sa, None)

        loss = b_xent(logits, lbl)
        losses.append(loss.item())
        # print("loss: ", loss)
        loss.backward()
        optimiser.step()


        if loss < best:
            best = loss
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            break

    model.load_state_dict(torch.load('model.pkl'))



    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    features = features.cuda()
    adj = adj.cuda()

    embeds, a = model.embed(features, adj, None)
    train_embs = embeds[0, idx_train]
    test_embs = embeds[0, idx_test]

    train_lbls = labels[idx_train]
    test_lbls = labels[idx_test]

    accs = []
    wd = 0.01 if dataset == 'citeseer' else 0.0

    for i in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
        log.cuda()
        for j in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    accs = torch.stack(accs)
    np.savetxt(osp.join("embeddings", args.dataset + str(accs.mean().item()) + ".txt"), embeds[0].detach().cpu().numpy())
    np.savetxt(osp.join("losses", args.dataset + str(accs.mean().item()) + ".txt"), np.array(losses))
    print("Accuracy:",accs.mean().item())


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    torch.cuda.set_device(1)        # choose gpu
    args = get_args()

    dataset = args.dataset
    for i in range(50):
        train(dataset, args)
