#!/usr/bin/env python


from src.RFF import RFF
from src.DManager import DManager
from src.terminal_print import *
from src.basic_optimizer import basic_optimizer
from src.line_plot import line_plot


import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
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
		self.phi= RFF(rff_width=db['RFF_Width'])		# RFF width

		self.W1 = torch.nn.Parameter(torch.randn((db['d'], db['RFF_Width']), device=db['device'] ), requires_grad=True)
		self.W2 = torch.nn.Parameter(torch.randn((db['RFF_Width'], db['RFF_Width']), device=db['device'] ), requires_grad=True)
		self.W3 = torch.nn.Parameter(torch.randn((db['RFF_Width'], db['out_dim']), device=db['device'] ), requires_grad=True)

		#self.output_network()

	def output_network(self):
		for name, W in self.named_parameters(): print(name, W.shape)

	def get_optimizer(self):
		return torch.optim.Adam(self.parameters(), lr=self.db['learning_rate'])
	
	def optimization_initialization(self):
		pass

	def on_new_epoch(self, loss, epoch, lr):
		self.loss_history.append(loss)
		write_to_current_line('loss: %.6f, epoch: %d, lr:%.7f'%(loss, epoch, lr))

	def predict(self, x):
		db = self.db

		# Use Relu
		y1 = torch.matmul(x, self.W1)
		phi_y1 = self.phi(y1)

		y2 = torch.matmul(phi_y1, self.W2)
		phi_y2 = self.phi(y2)

		y_hat = torch.matmul(phi_y2, self.W3)
		y_hat = F.softmax(y_hat, dim=1)
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

	N = 1000
	X1 = np.random.rand(N,3)
	X2 = np.random.rand(N,3) #+ 10
	X = np.vstack((X1,X2))
	Y = np.hstack((np.ones(N), np.zeros(N)))

	db = {}
	# db['loss_function'] = torch.nn.CrossEntropyLoss()			# torch.nn.functional.cross_entropy, torch.nn.MSELoss, torch.nn.CrossEntropyLoss
	db['loss_function'] = torch.nn.MSELoss()
	db['d'] = X.shape[1]
	db['RFF_Width'] = 100
	db['depth'] = 4
	db['device'] = 'cpu'
	db['out_dim'] = 1				# 1 if regression
	db['max_ℓ#'] = 200
	db['learning_rate'] = 0.001
	db['dataType'] = torch.FloatTensor

	DM = DManager(X, Y, db['dataType'])
	loader = DataLoader(dataset=DM, batch_size=16, shuffle=True, pin_memory=True, drop_last=True)

	#	Running RFF
	R = rff_net(db)
	basic_optimizer(R, loader)
	y_hat = R.predict(DM.X_Var)
	y_hat = np.argmax(y_hat.cpu().detach().numpy(), axis=1)
	Acc1 = accuracy_score(y_hat, Y)
	Lrff = np.array(R.loss_history)

	#	Running Relu
	R = rff_net(db)
	R.phi = F.relu
	basic_optimizer(R, loader)
	y_hat2 = R.predict(DM.X_Var)
	y_hat2 = np.argmax(y_hat2.cpu().detach().numpy(), axis=1)
	Acc2 = accuracy_score(y_hat2, Y)
	Lrelu = np.array(R.loss_history)


	#	Running sigmoid
	R = rff_net(db)
	R.phi = F.sigmoid
	basic_optimizer(R, loader)
	y_hat3 = R.predict(DM.X_Var)
	y_hat3 = np.argmax(y_hat3.cpu().detach().numpy(), axis=1)
	Acc3 = accuracy_score(y_hat3, Y)
	Lsigmoid = np.array(R.loss_history)


	#	Running tanh
	R = rff_net(db)
	R.phi = F.tanh
	basic_optimizer(R, loader)
	y_hat4 = R.predict(DM.X_Var)
	y_hat4 = np.argmax(y_hat4.cpu().detach().numpy(), axis=1)
	Acc4 = accuracy_score(y_hat4, Y)
	Ltanh = np.array(R.loss_history)


	epochs = np.arange(Lrff.shape[0])
	joint_loss = np.hstack((Lrelu, Lrff, Lsigmoid))

	LP = line_plot()
	LP.add_plot(epochs,Lrff, color='blue')
	LP.add_plot(epochs,Lrelu, color='green')
	LP.add_plot(epochs,Lsigmoid, color='red')
	LP.add_plot(epochs,Ltanh, color='yellow')
	LP.set_xlabel('Number of Epoch')
	LP.set_ylabel('Loss')
	LP.set_title('RFF Vs Relu Activation Functions')
	LP.add_text(epochs, joint_loss, 'Green:Relu, Acc:%.3f\n'
									'Red:Sigmoid, Acc:%.3f\n'
									'Orange:Tanh, Acc:%.3f\n'
									'Blue:RFF, Acc:%.3f'%
				(Acc2, Acc3, Acc4, Acc1), alpha=0.4, beta=0.5)
	LP.show()
	import pdb; pdb.set_trace()
