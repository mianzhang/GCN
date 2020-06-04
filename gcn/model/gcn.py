import torch
import torch.nn as nn

from .graphconvolve import GraphConvolve


class GCN(nn.Module):

    def __init__(self, input_size, out_size, A_norm, hidden_size):
        super(GCN, self).__init__()
        self.A_norm = nn.Parameter(A_norm, requires_grad=False)
        self.gc1 = GraphConvolve(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.gc2 = GraphConvolve(hidden_size, out_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.nll_loss = nn.NLLLoss()

    def get_prob(self, X):
        hidden = self.relu(self.gc1(self.A_norm, X))
        logP = self.logsoftmax(self.gc2(self.A_norm, hidden))
        return logP

    def forward(self, X, idxs):
        prob = self.get_prob(X)
        return torch.argmax(prob[idxs], dim=-1)

    def calculate_loss(self, X, idxs, tags):
        prob = self.get_prob(X)
        prob = prob[idxs]
        loss = self.nll_loss(prob, tags)
        return loss
