import torch.nn as nn



class PEM(nn.Module):
    def __init__(self, hiddensize):
        super(PEM, self).__init__()
        self.fc1 = nn.Linear(in_features = 32, out_features = hiddensize)
        self.fc2 = nn.Linear(in_features = hiddensize, out_features = 1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_features):
        fc1 = 0.1 * self.fc1(X_features)
        relu1 = self.relu(fc1)
        fc2 = 0.1 * self.fc2(relu1)
        output = self.sigmoid(fc2)

        return output


