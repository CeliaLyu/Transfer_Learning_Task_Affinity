import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.autograd import Variable
import os
import dataframe_image as dfi

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
    Neural net
'''

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim, num_dir, rnn_module):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.nd = num_dir

        if self.nd == 2:
            bidir = True
        else:
            bidir = False
        if rnn_module == "srnn":
            self.rnnmodule = nn.RNN(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=bidir)
        elif rnn_module == "lstm":
            self.rnnmodule = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=bidir)
        self.linear = nn.Linear(self.hidden_dim * self.nd, self.out_dim)

    def forward(self, x):
        out, _ = self.rnnmodule(x)
        out = self.linear(out[-1])
        out = out * 100
        return out


'''
    Load data, prediction, train, test
'''


def Windowed_Dataset(series, window_size, stride, batch_size, shuffle):
    """
    params:
        series: time series data
        window_size: K points in this window are used to predict the next (K+1) point
        stride: stride between windows
        batch_size: batch size for training
    return:
        ds_loader: wrap windowed data into pytorch dataloader
    """
    f_s = window_size + 1
    l = len(series)
    ds = torch.from_numpy(series)
    ds = torch.unsqueeze(ds, dim=1)
    ds = [ds[i:i+f_s] for i in range(0, l, stride) if i <= l-f_s]  # each chunk contains k+1 points, the last one is the target
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return ds_loader


def prediction(model_name, series, window_size):
    """
    params:
        series: time series data
        window_size: K points in this window are used to predict the next (K+1) point
        model_name: trained model used for prediction
    return:
        result: predicted value of the series starting from (window_size+1) point
    """
    model = torch.load(model_name)
    model.eval()
    model = model.to(device)
    forcast = []

    series_t = torch.tensor(series)
    series_t = torch.unsqueeze(series_t, dim=1)
    for time_step in range(len(series) - window_size):
        Input_time = series_t[time_step:time_step+window_size,:]
        Input_time = Input_time.float().to(device)
        forcast.append(model(Input_time))
    result = forcast
    result = [x.detach().cpu().numpy().squeeze() for x in result]
    return result


def make_optimizer(optimizer_name, model, **kwargs):
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=kwargs['lr'], momentum=kwargs.get('momentum', 0.),
                              weight_decay=kwargs.get('weight_decay', 0.))
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(scheduler_name, optimizer, **kwargs):
    if scheduler_name == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs['milestones'], gamma=kwargs['factor'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def train(model, train_loader, batch_size_train, optimizer, criterion, scheduler):

    model.train()
    running_train_loss = 0

    for batch_index, item in enumerate(train_loader):
        """
        item shape: batch_size * (window_size+1) * 1
        inputs shape: batch_size * (window_size) * 1
        target(last point in item) shape: batch_size * 1
        """
        inputs = item[0:batch_size_train, 0:-1]
        inputs = torch.transpose(inputs, 0, 1)
        inputs = inputs.float().to(device)
        target = item[0:batch_size_train, -1:].squeeze(dim=1)
        target = target.float().to(device)

        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_train_loss += loss.item() * len(item)
    return running_train_loss / len(train_loader.dataset)


def test(model, test_loader, batch_size_test, criterion):

    model.eval()
    running_test_loss = 0

    with torch.no_grad():
        for batch_index, item in enumerate(test_loader):
            """
            item shape: batch_size * (window_size+1) * 1
            inputs shape: batch_size * (window_size) * 1
            target(last point in item) shape: batch_size * 1
            """
            inputs_test = item[0:batch_size_test, 0:-1]
            inputs_test = torch.transpose(inputs_test, 0, 1)
            inputs_test = inputs_test.float().to(device)
            target_test = item[0:batch_size_test, -1:].squeeze(dim=1)
            target_test = target_test.float().to(device)

            out_test = model(inputs_test)
            loss_test = criterion(out_test, target_test)
            running_test_loss += loss_test.item() * len(item)
    return running_test_loss / len(test_loader.dataset)


'''
    Fisher distance
'''


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


def diag_fisher(model, data, batch_size_test):
    '''
    model is from base task, data is from target task.
    data: dataloader form
    '''
    precision_matrices = {}
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        precision_matrices[n] = variable(p.data)

    model.eval()

    # loss function during testing/computing the Fisher Information matrices
    criterion = nn.MSELoss()

    for batch_index, item in enumerate(data):
        inputs = item[0:batch_size_test, 0:-1]
        inputs = torch.transpose(inputs, 0, 1)
        inputs = inputs.float().to(device)
        target = item[0:batch_size_test, -1:].squeeze(dim=1)
        target = target.float().to(device)

        model.zero_grad()
        out = model(inputs)
        loss = criterion(out, target)
        loss.backward()

        # Compute the Fisher Information as (p.grad.data ** 2)
        for n, p in model.named_parameters():
            precision_matrices[n].data += (p.grad.data ** 2).mean(0)  

    # Fisher Information
    precision_matrices = {n: p for n, p in precision_matrices.items()}

    return precision_matrices


def Fisher_distance(fisher_matrix_source, fisher_matrix_target, model):
    '''
    fisher_matrix_source: output of diag_fisher(model_base, data_base)
    fisher_matrix_target: output of diag_fisher(model_base, data_target)
    model: neural network trained on the source
    Fisher_distance: return the square of distance
    '''
    distance = 0
    for n, p in model.named_parameters():
        distance += 0.5 * np.sum(((fisher_matrix_source[n] ** 0.5 - fisher_matrix_target[n] ** 0.5) ** 2).cpu().numpy())
    return distance


def compute_distance(model, source_task, target_task, batch_size_test_source, batch_size_test_target):
    '''
    model: trained neural network
    source_task: the data on which the neural network was trained
    target_task: the data that we want to estimate how close is it to the original source task
    '''

    fisher_source = diag_fisher(model, source_task, batch_size_test_source)
    fisher_target = diag_fisher(model, target_task, batch_size_test_target)

    distance = Fisher_distance(fisher_source, fisher_target, model)
    return distance

