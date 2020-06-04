from tqdm import tqdm
import torch

import gcn

log = gcn.utils.get_logger()


class Graph:

    def __init__(self, node_size):
        self.node_size = node_size
        self.edges = {}

    def add_edge(self, x, y, v):
        self.edges[(x, y)] = v

    def get_edge_size(self):
        return len(self.edges)

    def get_node_size(self):
        return self.node_size

    def get_adj_norm(self, device):
        A = torch.zeros((self.node_size, self.node_size)).to(device)
        xis = [xi for (xi, _), _ in self.edges.items()]
        yis = [yi for (_, yi), _ in self.edges.items()]
        vs = [v for (_, _), v in self.edges.items()]
        k = torch.sum(torch.tensor(vs).to(device) == 0).item()
        log.info("{} / {} 0s in edges".format(k, len(vs)))
        A[xis, yis] = torch.tensor(vs).to(device)
        A = A + A.transpose(0, 1) + torch.eye(self.node_size).to(device)
        zeros = torch.sum(A == 0).item()
        log.info("{} / {} 0s in A".format(zeros, self.node_size * self.node_size))

        D = torch.diag(torch.sum(A != 0, dim=0)).float()
        D_ = torch.sqrt(torch.inverse(D))
        return torch.mm(torch.mm(D_, A), D_).to("cpu")

