import torch.nn as nn
import torch.nn.functional as F
from gcn.pygcn.layers import GraphConvolution
import torch
import numpy as np
from torch.autograd import Variable

class rff(nn.Module):
    # rff_width, the larger the better approximation
    def __init__(self, sigma=1, rff_width=20000):
        super(rff, self).__init__()
        self.sigma = sigma
        self.rff_width = rff_width
        self.phase_shift = None
        self.RFF_scale = np.sqrt(2.0 / self.rff_width)

    def initialize_RFF(self, x, sigma, output_torch, dtype=torch.FloatTensor):
        self.x = x

        if self.phase_shift is not None: return
        # if x.shape[0] == self.N: return

        self.N = x.shape[0]
        self.d = x.shape[1]
        self.sigma = sigma

        if type(x) == torch.Tensor or type(x) == np.ndarray:
            self.phase_shift = 2 * np.pi * np.random.rand(1, self.rff_width)
            # self.phase_shift = np.matlib.repmat(b, self.N, 1)
            self.rand_proj = np.random.randn(self.d, self.rff_width) / (self.sigma)
        else:
            raise ValueError('An unknown datatype is passed into get_rbf as %s' % str(type(x)))

        self.use_torch(output_torch, dtype)

    def use_torch(self, output_torch, dtype):
        if not output_torch: return

        dvc = self.x.device
        self.phase_shift = torch.from_numpy(self.phase_shift)
        self.phase_shift = Variable(self.phase_shift.type(dtype), requires_grad=False)
        self.phase_shift = self.phase_shift.to(dvc,
                                               non_blocking=True)  # make sure the data is stored in CPU or GPU device

        self.rand_proj = torch.from_numpy(self.rand_proj)
        self.rand_proj = Variable(self.rand_proj.type(dtype), requires_grad=False)
        self.rand_proj = self.rand_proj.to(dvc, non_blocking=True)  # make sure the data is stored in CPU or GPU device

    def forward(self, x):
        self.initialize_RFF(x, self.sigma, True)

        if type(self.x) == np.ndarray:
            self.x = torch.from_numpy(self.x)
            self.x = Variable(self.x.type(self.dtype), requires_grad=False)

        elif type(self.x) != torch.Tensor:
            raise ValueError('An unknown datatype is passed into get_rbf as %s' % str(type(self.x)))

        P = self.RFF_scale * torch.cos(torch.mm(self.x, self.rand_proj) + self.phase_shift)
        return P

    def torch_rbf(self, x, sigma):
        P = self.__call__(x, sigma)
        K = torch.mm(P, P.transpose(0, 1))
        # K = (2.0/self.rff_width)*K
        K = F.relu(K)

        return K

    def np_feature_map(self, x):
        const = np.sqrt(2.0 / self.rff_width)
        feature_map = const * np.cos(x.dot(self.rand_proj) + self.phase_shift)

        return feature_map

    def np_rbf(self):
        P = np.cos(self.x.dot(self.rand_proj) + self.phase_shift)
        K = (2.0 / self.rff_width) * (P.dot(P.T))
        K = np.maximum(K, 0)
        K = np.minimum(K, 1)
        return K

    def get_rbf(self, x, sigma, output_torch=False, dtype=torch.FloatTensor):
        self.dtype = dtype
        self.initialize_RFF(x, sigma, output_torch, dtype)

        if output_torch:
            return self.torch_rbf(x, sigma)
        else:
            return self.np_rbf()

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nhid)
        # self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.act = rff(rff_width=nhid)
        # self.act = F.relu

    def forward(self, x, adj):
        x = self.act(self.gc1(x, adj))
        # x = self.act(self.gc2(x, adj))
        # x = self.act(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        return F.log_softmax(x, dim=1)
