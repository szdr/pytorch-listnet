from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self, D):
        super(Net, self).__init__()
        self.l1 = nn.Linear(D, 10)
        self.l2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = self.l2(x)
        return x


def listnet_loss(y_i, z_i):
    """
    y_i: (n_i, 1)
    z_i: (n_i, 1)
    """

    P_y_i = F.softmax(y_i, dim=0)
    P_z_i = F.softmax(z_i, dim=0)
    return - torch.sum(P_y_i * torch.log(P_z_i))


def make_dataset(N_train, N_valid, D):
    ws = torch.randn(D, 1)

    X_train = torch.randn(N_train, D, requires_grad=True)
    X_valid = torch.randn(N_valid, D, requires_grad=True)

    ys_train_score = torch.mm(X_train, ws)
    ys_valid_score = torch.mm(X_valid, ws)

    bins = [-2, -1, 0, 1]  # 5 relevances
    ys_train_rel = torch.Tensor(
        np.digitize(ys_train_score.clone().detach().numpy(), bins=bins)
    )
    ys_valid_rel = torch.Tensor(
        np.digitize(ys_valid_score.clone().detach().numpy(), bins=bins)
    )

    return X_train, X_valid, ys_train_rel, ys_valid_rel


def swapped_pairs(ys_pred, ys_target):
    N = ys_target.shape[0]
    swapped = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            if ys_target[i] < ys_target[j]:
                if ys_pred[i] > ys_pred[j]:
                    swapped += 1
            elif ys_target[i] > ys_target[j]:
                if ys_pred[i] < ys_pred[j]:
                    swapped += 1
    return swapped


def ndcg(ys_true, ys_pred):
    def dcg(ys_true, ys_pred):
        _, argsort = torch.sort(ys_pred, descending=True, dim=0)
        ys_true_sorted = ys_true[argsort]
        ret = 0
        for i, l in enumerate(ys_true_sorted, 1):
            ret += (2 ** l - 1) / np.log2(1 + i)
        return ret
    ideal_dcg = dcg(ys_true, ys_true)
    pred_dcg = dcg(ys_true, ys_pred)
    return pred_dcg / ideal_dcg


if __name__ == '__main__':
    N_train = 500
    N_valid = 100
    D = 50
    epochs = 10
    batch_size = 16

    X_train, X_valid, ys_train, ys_valid = make_dataset(N_train, N_valid, D)

    net = Net(D)
    opt = optim.Adam(net.parameters())

    for epoch in range(epochs):
        idx = torch.randperm(N_train)

        X_train = X_train[idx]
        ys_train = ys_train[idx]

        cur_batch = 0
        for it in range(N_train // batch_size):
            batch_X = X_train[cur_batch: cur_batch + batch_size]
            batch_ys = ys_train[cur_batch: cur_batch + batch_size]
            cur_batch += batch_size

            opt.zero_grad()
            if len(batch_X) > 0:
                batch_pred = net(batch_X)
                batch_loss = listnet_loss(batch_ys, batch_pred)
                batch_loss.backward(retain_graph=True)
                opt.step()

        with torch.no_grad():
            valid_pred = net(X_valid)
            valid_swapped_pairs = swapped_pairs(valid_pred, ys_valid)
            ndcg_score = ndcg(ys_valid, valid_pred).item()
            print(f"epoch: {epoch + 1} valid swapped pairs: {valid_swapped_pairs}/{N_valid * (N_valid - 1) // 2} ndcg: {ndcg_score:.4f}")
