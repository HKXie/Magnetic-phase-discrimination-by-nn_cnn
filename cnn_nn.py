import torch
import torchvision
from torch import nn
import time
import torch.nn.functional as F
from torch import optim
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader


import numpy as np
import matplotlib.pylab as plt

class CNN(nn.Module):

    def __init__(self, configs):
        # initialise parent pytorch class
        super().__init__()

        self.config = configs

        # define neural network layers
        self.convseq = nn.Sequential(

            nn.Conv2d(self.config.channels, 28, kernel_size=4, stride=2),
            nn.BatchNorm2d(28),
            nn.Dropout(p=self.config.dropout_rate),

            nn.LeakyReLU(),  # 8*8
            # nn.GELU(),

            nn.Conv2d(28, 28, kernel_size=3, stride=1),
            nn.BatchNorm2d(28),
            nn.Dropout(p=self.config.dropout_rate),
            nn.LeakyReLU(),  # 6*6
            # nn.GELU(),

            nn.Conv2d(28, 28, kernel_size=3, stride=1),
            nn.BatchNorm2d(28),
            nn.Dropout(p=self.config.dropout_rate),
            nn.LeakyReLU(),  # 6*6

            nn.Conv2d(28, 28, kernel_size=3, stride=1),
            nn.BatchNorm2d(28),
            nn.Dropout(p=self.config.dropout_rate),
            nn.LeakyReLU(),  # 6*6

        )

        self.dense = nn.Sequential(
            nn.Linear(28 * 3 * 3, self.config.h_dim),
            nn.Dropout(p=self.config.dropout_rate),

            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.LayerNorm(self.config.h_dim),
            nn.BatchNorm1d(self.config.h_dim),

            nn.Linear(self.config.h_dim, self.config.class_num),
            # nn.Sigmoid()
            # nn.Softmax()

        )

    def forward(self, inputs):
        outputs = self.convseq(inputs)
        outputs = outputs.view(outputs.size(0), -1)  # or use [ outputs.size(0), ]
        outputs = self.dense(outputs)
        return outputs


class NN(nn.Module):

    def __init__(self, configs,):
        # initialise parent pytorch class
        super().__init__()
        self.configs = configs

        layers = [nn.Linear(self.configs.v_dim, self.configs.h_dim), nn.ReLU(), nn.Dropout(p=self.configs.dropout_rate), ]

        for _ in range(self.configs.h_depth):
            layers.append(nn.Linear(self.configs.h_dim, self.configs.h_dim))
            layers.append(nn.LayerNorm(self.configs.h_dim))

            layers.append(nn.ReLU())
            # layers.append(nn.LayerNorm(self.configs.h_dim))
            layers.append(nn.Dropout(p=self.configs.dropout_rate))

        layers.append(nn.Linear(self.configs.h_dim, self.configs.class_num))

        # self.model = nn.Sequential(*layers)
        self.model = nn.ModuleList(layers)

    def forward(self, x):

        # # simply run model
        # return self.model(x)

        for layer in self.model:
            x = layer(x)

        return x