import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn


class TEM(nn.Module):

    def __init__(self, embedsize=64, hiddensize=128):
        super(TEM, self).__init__()
        batchNormalization = False
        self.linear = nn.Linear(in_features=1024, out_features=embedsize)
        self.tmp_conv0 = nn.Conv1d(in_channels=embedsize, out_channels=hiddensize, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.tmp_conv1 = nn.Conv1d(in_channels=hiddensize, out_channels=hiddensize, kernel_size=3, stride=1, padding=1)
        self.tmp_conv2 = nn.Conv1d(in_channels=hiddensize, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # tem = nn.Sequential()
        # tem.add_module('conv1d{0}'.format(0),
        #                nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1))
        # tem.add_module('relu{0}'.format(0), nn.ReLU(True))
        #
        # tem.add_module('conv1d{0}'.format(1),
        #                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        # tem.add_module('relu{0}'.format(1), nn.ReLU(True))
        #
        # tem.add_module('conv1d{0}'.format(2),
        #                nn.Conv1d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding=0))
        # self.cnn = tem
        #
        # self.sigmoid = nn.Sigmoid()

    def forward(self, X_feature):
        conv = self.cnn(X_feature)
        sigmoid_output = self.sigmoid(0.1 * conv)
        return sigmoid_output