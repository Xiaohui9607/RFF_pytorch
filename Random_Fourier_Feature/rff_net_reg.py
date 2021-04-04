#!/usr/bin/env python


from src.RFF import RFF
from src.DManager import DManager
from src.terminal_print import *
from src.basic_optimizer import basic_optimizer
from src.line_plot import line_plot

import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import time


class rff_net(torch.nn.Module):
    def __init__(self, db, learning_rate=0.001):
        super(rff_net, self).__init__()
        self.db = db
        self.loss_history = []
        self.learning_rate = learning_rate
        self.phi = RFF(rff_width=db['RFF_Width'])  # RFF width
        self.linear1 = torch.nn.Linear(db['d'], db['RFF_Width'], bias=True)
        self.linear2 = torch.nn.Linear(db['RFF_Width'], db['RFF_Width'], bias=True)
        self.linear3 = torch.nn.Linear(db['RFF_Width'], db['out_dim'], bias=True)
        # self.W1 = torch.nn.Parameter(torch.randn((db['d'], db['RFF_Width']), device=db['device']), requires_grad=True)
        # self.W2 = torch.nn.Parameter(torch.randn((db['RFF_Width'], db['RFF_Width']), device=db['device']),
        #                              requires_grad=True)
        # self.W3 = torch.nn.Parameter(torch.randn((db['RFF_Width'], db['out_dim']), device=db['device']),
        #                              requires_grad=True)

    # self.output_network()

    def output_network(self):
        for name, W in self.named_parameters(): print(name, W.shape)

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.db['learning_rate'])

    def optimization_initialization(self):
        pass

    def on_new_epoch(self, loss, epoch, lr):
        self.loss_history.append(loss)
        write_to_current_line('loss: %.6f, epoch: %d, lr:%.7f' % (loss, epoch, lr))

    def predict(self, x):
        db = self.db

        # Use Relu
        x = self.linear1(x)
        phi_y1 = self.phi(x)

        x = self.linear2(phi_y1)
        phi_y2 = self.phi(x)

        y_hat = self.linear3(phi_y2)
        # y_hat = F.softmax(y_hat, dim=1)
        return y_hat

    def forward(self, x, y, i):
        db = self.db
        y_hat = self.predict(x)

        if type(db['loss_function']) == type(torch.nn.CrossEntropyLoss()):
            y = y.to(device=db['device'], dtype=torch.int64)

        loss = db['loss_function'](y_hat, y)
        return loss


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=300)
    np.set_printoptions(suppress=True)
    torch.set_printoptions(edgeitems=3)
    torch.set_printoptions(threshold=10_00)
    torch.set_printoptions(linewidth=400)

    N = 6000
    X1 = np.random.rand(N,1)
    X2 = np.random.rand(N,1)
    X3 = np.random.rand(N,1)
    X4 = np.random.rand(N,1)
    X5 = np.random.rand(N,1)
    # c = np.random.rand(9)
    ploy1 = np.polynomial.polynomial.Polynomial(np.random.rand(9))
    ploy3 = np.polynomial.polynomial.Polynomial(np.random.rand(7))
    ploy4 = np.polynomial.polynomial.Polynomial(np.random.rand(8))
    ploy2 = np.polynomial.polynomial.Polynomial(np.random.rand(3))
    Y = ploy1(X1) + np.sin(X2) + ploy2(X3) + ploy3(X4)+ ploy4(X5)
    X = np.hstack((X1, X2, X3, X4, X5))
    # X = X[:,np.newaxis]
    X_train = X[:N//3*2]
    X_val = X[N//3*2:]+(1+np.random.rand(N//3,1))*0.01
    Y_train = Y[:N//3*2]#+ np.random.randn(N//3*2,1)*0.01
    Y_val = Y[N//3*2:]

    db = {}
    # db['loss_function'] = torch.nn.CrossEntropyLoss()			# torch.nn.functional.cross_entropy, torch.nn.MSELoss, torch.nn.CrossEntropyLoss
    db['loss_function'] = torch.nn.MSELoss(reduce='l2')
    db['d'] = X.shape[1]
    db['RFF_Width'] = 4
    db['depth'] = 4
    db['device'] = 'cpu'
    db['out_dim'] = 1  # 1 if regression
    db['max_ℓ#'] = 400
    db['learning_rate'] = 0.001
    db['dataType'] = torch.FloatTensor

    DM_train = DManager(X_train, Y_train, db['dataType'])
    loader_train = DataLoader(dataset=DM_train, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)
    DM_val = DManager(X_val, Y_val, db['dataType'])
    loader_val = DataLoader(dataset=DM_val, batch_size=32, shuffle=False, pin_memory=True, drop_last=True)

    # Running RFF
    R = rff_net(db)
    basic_optimizer(R, loader_train)
    y_hat = R.predict(DM_val.X_Var).detach().numpy()
    Acc1 = mean_squared_error(y_hat, Y_val)
    Lrff = np.array(R.loss_history)

    # Running Relu
    R = rff_net(db)
    R.phi = F.relu
    basic_optimizer(R, loader_train)
    y_hat2 = R.predict(DM_val.X_Var).cpu().detach().numpy()
    # y_hat2 = np.argmax(y_hat2.cpu().detach().numpy(), axis=1)
    Acc2 = mean_squared_error(y_hat2, Y_val)
    Lrelu = np.array(R.loss_history)

    # # Running sigmoid
    # R = rff_net(db)
    # R.phi = F.sigmoid
    # basic_optimizer(R, loader_train)
    # y_hat3 = R.predict(DM_val.X_Var).cpu().detach().numpy()
    # # y_hat3 = np.argmax(y_hat3.cpu().detach().numpy(), axis=1)
    # Acc3 = mean_squared_error(y_hat3, Y_val)
    # Lsigmoid = np.array(R.loss_history)
    #
    # # Running tanh
    # R = rff_net(db)
    # R.phi = F.tanh
    # basic_optimizer(R, loader_train)
    # y_hat4 = R.predict(DM_val.X_Var).cpu().detach().numpy()
    # # y_hat4 = np.argmax(y_hat4.cpu().detach().numpy(), axis=1)
    # Acc4 = mean_squared_error(y_hat4, Y_val)
    # Ltanh = np.array(R.loss_history)

    epochs = np.arange(Lrff.shape[0])
    joint_loss = np.hstack((Lrelu, Lrff))

    import matplotlib.pyplot as plt

    # plt.scatter(X_val,Y_val)
    # plt.scatter(X_val,y_hat, label="rff")
    # plt.scatter(X_val,y_hat2, label="relu")
    # plt.scatter(X_val,y_hat3, label="sigmoid")
    # plt.scatter(X_val,y_hat4, label="tanh")
    #
    # plt.show()

    #
    LP = line_plot()

    LP.add_plot(epochs, Lrff, color='blue')
    LP.add_plot(epochs, Lrelu, color='green')
    # LP.add_plot(epochs, Lsigmoid, color='red')
    # LP.add_plot(epochs, Ltanh, color='yellow')
    LP.set_xlabel('Number of Epoch')
    LP.set_ylabel('Loss')
    LP.set_title('RFF Vs Relu Activation Functions')
    LP.add_text(epochs, joint_loss, 'Green:Relu, MSE:%.3f\n'
                                    'Blue:RFF, MSE:%.3f' %
                                    # 'Red:Sigmoid, MSE:%.3f\n'
                                    # 'Orange:Tanh, MSE:%.3f\n'
                (Acc2, Acc1), alpha=0.4, beta=0.5)
    LP.show()

    import pdb;pdb.set_trace()
