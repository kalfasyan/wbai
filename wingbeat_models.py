import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv1dNetRAW(nn.Module):
    def __init__(self):
        super(Conv1dNetRAW, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, 3)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, 3)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(64, 128, 3)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(128, 256, 3)
        self.bn5 = nn.BatchNorm1d(256)
        self.pool5 = nn.MaxPool1d(2)    

        self.dropout = nn.Dropout()
        self.avgPool = nn.AvgPool1d(154)
        self.fc1 = nn.Linear(256, 2)

    def forward(self, x):
        # print("######")

        x = self.conv1(x)
        # print(f"conv1: {x.shape}")
        x = F.relu(self.bn1(x))
        # print(f"relu2: {x.shape}")
        x = self.pool1(x)
        # print(f"pool1: {x.shape}")

        x = self.conv2(x)
        # print(f"conv2: {x.shape}")
        x = F.relu(self.bn2(x))
        # print(f"relu2: {x.shape}")
        x = self.pool2(x)
        # print(f"pool2: {x.shape}")

        x = self.conv3(x)
        # print(f"conv3: {x.shape}")
        x = F.relu(self.bn3(x))
        # print(f"relu3: {x.shape}")
        x = self.pool3(x)
        # print(f"pool3: {x.shape}")

        x = self.conv4(x)
        # print(f"conv4: {x.shape}")
        x = F.relu(self.bn4(x))
        # print(f"relu4: {x.shape}")
        x = self.pool4(x)
        # print(f"pool4: {x.shape}")

        x = self.conv5(x)
        # print(f"conv5: {x.shape}")
        x = F.relu(self.bn5(x))
        # print(f"relu5: {x.shape}")
        x = self.pool5(x)
        # print(f"pool5: {x.shape}")

        x = self.dropout(x)
        x = self.avgPool(x)
        # print(f"avgPool: {x.shape}")
        # x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = x.view(x.shape[0], -1)
        # print(f"view: {x.shape}")
        x = self.fc1(x)
        # print(f"fc1: {x.shape}")
        return x

class Conv1dNetPSD(nn.Module):
    def __init__(self):
        super(Conv1dNetPSD, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, 3)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, 3)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(64, 128, 3)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2) 

        self.dropout = nn.Dropout()
        self.avgPool = nn.AvgPool1d(127)
        self.fc1 = nn.Linear(256, 2)

    def forward(self, x):
        # print("######")

        x = self.conv1(x)
        # print(f"conv1: {x.shape}")
        x = F.relu(self.bn1(x))
        # print(f"relu2: {x.shape}")
        x = self.pool1(x)
        # print(f"pool1: {x.shape}")

        x = self.conv2(x)
        # print(f"conv2: {x.shape}")
        x = F.relu(self.bn2(x))
        # print(f"relu2: {x.shape}")
        x = self.pool2(x)
        # print(f"pool2: {x.shape}")

        x = self.conv3(x)
        # print(f"conv3: {x.shape}")
        x = F.relu(self.bn3(x))
        # print(f"relu3: {x.shape}")
        x = self.pool3(x)
        # print(f"pool3: {x.shape}")

        x = self.conv4(x)
        # print(f"conv4: {x.shape}")
        x = F.relu(self.bn4(x))
        # print(f"relu4: {x.shape}")
        x = self.pool4(x)
        # print(f"pool4: {x.shape}")

        # x = self.conv5(x)
        # print(f"conv5: {x.shape}")
        # x = F.relu(self.bn5(x))
        # print(f"relu5: {x.shape}")
        # x = self.pool5(x)
        # print(f"pool5: {x.shape}")

        x = self.dropout(x)
        x = self.avgPool(x)
        # print(f"avgPool: {x.shape}")
        # x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = x.view(x.shape[0], -1)
        # print(f"view: {x.shape}")
        x = self.fc1(x)
        # print(f"fc1: {x.shape}")
        # x = F.log_softmax(x, dim = 1)
        # print(f"log_softmax: {x.shape}")
        return x

# Credit: Bjarte Mehus Sunde from Github
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='data_created/checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss