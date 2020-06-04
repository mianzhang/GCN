import torch
import torch.nn as nn


class GraphConvolve(nn.Module):

    def __init__(self, input_size, output_size, bias=False):
        super(GraphConvolve, self).__init__()
        self.weight = nn.Parameter(torch.zeros(input_size, output_size, dtype=torch.float),
                                   requires_grad=True)
        var = 2. / (self.weight.size(0) + self.weight.size(1))
        self.weight.data.normal_(0, var)
        self.drop = nn.Dropout(p=0.2)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size, dtype=torch.float),
                                     requires_grad=True)
            self.bias.data.normal_(0, var)
        else:
            self.bias = None

    def forward(self, A_norm, X):
        out = torch.mm(self.drop(torch.mm(A_norm, X)), self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
