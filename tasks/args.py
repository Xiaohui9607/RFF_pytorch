import torch
import argparse
import os

class Args:
    def __init__(self):

        self.parser = argparse.ArgumentParser()
        # trials
        self.parser.add_argument('--act_func', type=str, default='relu', help='relu|rff|sigmoid|tanh')
        # self.parser.add_argument('--task', type=str, default='classification', help='classification|regression|...')
        self.parser.add_argument('--dataset', type=str, default='cifar-10', help='task+dataset must be a valid combination')
        self.parser.add_argument('--backbone', type=str, default='resnet18', help='mlp|resnet|...')
        self.parser.add_argument('--pretrained', type=bool, default=False, help='use pretrained weight provided by pytorch')
        # more..

        # trainings
        self.parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning_rate')
        self.parser.add_argument('--optimizer', type=str, default='SGD', help='SGD|Adam|...')
        self.parser.add_argument('--epoch', type=int, default=200, help='')
        self.parser.add_argument('--batchsize', type=int, default=32, help='')
        self.parser.add_argument('--seed', type=int, default=42, help='random seed')
        self.parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='cuda:[d] | cpu')
        # more.. (early stop, patient, etc.)


    def update_args(self):
        args = self.parser.parse_args()
        return args