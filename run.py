import torch
# import torchvision
from torch import nn
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from einops import rearrange
# from torchvision import datasets, transforms
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

import argparse

import numpy as np
import matplotlib.pylab as plt
import cnn_nn
from dataloader import data_provider
import train

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNN_NN_heisenberg_pahses_classification')

    parser.add_argument('--data_path', type=str, default='data',help='')
    parser.add_argument('--train_data_name', type=str,
                        default=['124_MT_20size_1T_jiangede.hdf5',],help='')

    parser.add_argument('--test_data_name', type=str,
                        default=['124_MT_20size_2T_jiangede.hdf5',],help='')

    parser.add_argument('--model_type', type=str, default='cnn', help='_')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--imgage-size', type=int, default=20,help='input image size')
    parser.add_argument('--imgage-patch', type=int, default=4,help='segmentation image patch')
    parser.add_argument('--h_dim', type=int, default=128,help='dim of MLP')
    parser.add_argument('--v_dim', type=int, default=800,help='dim of MLP')
    # parser.add_argument('--mlp-dim', type=int, default=128, help='_')
    parser.add_argument('--h_depth', type=int, default=2, help='depth of hidden layers')
    parser.add_argument('--channels', type=int, default=2, help='dim of MLP')
    parser.add_argument('--class_num', type=int, default=2,help='num class')
    parser.add_argument('--num_token',type=int, default=96, help='_')
    parser.add_argument('--dropout-rate', type=int, default=0.3, help='_')


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model_type == 'cnn':
        model_cnn = cnn_nn.CNN(args)
    else:
        model_cnn = cnn_nn.NN(args)

    exp = train.NN_train_task(model_cnn, args)
    exp.train_valid()



