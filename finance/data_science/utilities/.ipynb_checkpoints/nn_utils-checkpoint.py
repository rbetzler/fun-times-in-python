import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, functional, optim, autograd

from finance.utilities import utils
from finance.data_science.utilities import cluster_utils

    
class TorchNN(nn.Module):
    def __init__(self, 
                 train_x,
                 train_y,
                 test_x,
                 test_y,
                 n_layers=2,
                 bias=True,
                 dropout=0,
                 hidden_shape=16,
                 output_shape=1,
                 batch_size=1,
                 n_epochs=10,
                 learning_rate=.0001,
                 device='cuda'
                ):
        super(TorchNN, self).__init__()
                
        self.n_layers = n_layers
        self.input_shape = train_x.shape[1]
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device

        self.train_x = torch.tensor(train_x.values
                                   ).to(self.device).float().detach().requires_grad_(True)
        self.train_y = torch.tensor(train_y.values
                                   ).to(self.device).float()
        
        self.test_x = torch.tensor(test_x.values
                                   ).to(self.device).float().detach().requires_grad_(True)
        self.test_y = torch.tensor(test_y.values
                                   ).to(self.device).float()

        self.relu = nn.ReLU()
        self.linear_one = nn.Linear(self.input_shape, self.hidden_shape).to(self.device)
        self.linear_two = nn.Linear(self.hidden_shape, self.output_shape).to(self.device)

    def reset_network(self):
        self.linear_one.reset_parameters()
        self.linear_two.reset_parameters()
   
    @property
    def loss_function(self):
        # loss = nn.MSELoss(reduction='sum').to(self.device)
        loss = nn.L1Loss(reduction='sum').to(self.device)
        # loss = nn.KLDivLoss(reduction='sum').to(self.device)
        return loss
    
    @property
    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, train=True):
        if train:
            data = self.train_x.view(len(self.train_x), self.batch_size, -1)
        else: 
            data = self.test_x.view(len(self.test_x), self.batch_size, -1)

        output = self.linear_one(data)
        output = self.relu(output)
        output = self.linear_two(output)
        return output
    
    def execute(self):
#         optimizer = self.optimizer
        history = np.zeros(self.n_epochs)

        for epoch in range(self.n_epochs):
            self.zero_grad()

            prediction = self.forward()

            loss = self.loss_function(prediction, self.train_y)
            if epoch % int(self.n_epochs/10) == 0:
                print("Epoch ", epoch, "Error: ", loss.item())
            
            history[epoch] = loss.item()

#             optimizer.zero_grad()
            loss.backward()
#             optimizer.step()
    
    def predict(self):
        return self.forward(train=False)

    def plot_prediction_train(self):
        plt.title('Prediction v Actual - Train')
        plt.plot(self.train_y.cpu().numpy(), color='b')
        plt.plot(self.forward().view(-1).cpu().detach().numpy(), color='r')
        plt.show()
    
    def plot_prediction_train_error(self):
        plt.title('Prediction Error - Train')
        plt.plot(self.forward().view(-1).cpu().detach().numpy() 
                 - self.train_y.cpu().detach().numpy())
        plt.show()
    
    def plot_prediction_test(self):
        plt.title('Prediction v Actuals - Test')
        plt.plot(self.test_y.cpu().detach().numpy(), color='b')
        plt.plot(self.predict().view(-1).cpu().detach().numpy(), color='r')
        plt.show()
    
    def plot_prediction_test_error(self):
        plt.title('Prediction Error - Test')
        plt.plot(self.predict().view(-1).cpu().detach().numpy() 
                 - self.test_y.cpu().detach().numpy())
        plt.show()


class TorchLSTM(nn.Module):
    def __init__(self, 
                 train_x,
                 train_y,
                 test_x,
                 test_y,
                 n_layers=2,
                 bias=True,
                 dropout=0,
                 hidden_shape=16,
                 output_shape=1,
                 batch_size=1,
                 n_epochs=10,
                 learning_rate=.0001,
                 device='cuda'
                ):
        super(TorchLSTM, self).__init__()
                
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.input_shape = train_x.shape[1]
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.batch_size = len(train_x)
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.device = device

        self.train_x = torch.tensor(train_x.values
                                   ).to(self.device).float().detach().requires_grad_(True)
        self.train_y = torch.tensor(train_y.values
                                   ).to(self.device).float()
        
        self.test_x = torch.tensor(test_x.values
                                   ).to(self.device).float().detach().requires_grad_(True)
        self.test_y = torch.tensor(test_y.values
                                   ).to(self.device).float()

        self.lstm = nn.LSTM(input_size=self.input_shape, 
                            hidden_size=self.hidden_shape, 
                            num_layers=self.n_layers,
                            bias=self.bias,
                            dropout=self.dropout).to(self.device)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(self.hidden_shape, self.output_shape).to(self.device)
    
    def reset_network(self):
        self.lstm.reset_parameters()
        self.linear.reset_parameters()
   
    @property
    def loss_function(self):
        loss = nn.MSELoss(reduction='sum').to(self.device)
#         loss = nn.L1Loss(reduction='sum').to(self.device)
        # loss = nn.KLDivLoss(reduction='sum').to(self.device)
        return loss
    
    @property
    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def create_hidden_states(self):
        hidden = torch.randn(self.n_layers, 
                             self.batch_size, 
                             self.hidden_shape).to(self.device)

        cell = torch.randn(self.n_layers, 
                           self.batch_size,
                           self.hidden_shape).to(self.device)
        return hidden, cell

    def forward(self, train=True):
        if train:
            data = self.train_x.view(-1, self.batch_size, self.input_shape)
        else: 
            data = self.test_x.view(-1, self.batch_size, self.input_shape)

        output, self.hidden = self.lstm(data, self.hidden)
#         output = self.relu(output)
        output = self.linear(output)
        return output
    
    def execute(self):
        optimizer = self.optimizer
        history = np.zeros(self.n_epochs)

        for epoch in range(self.n_epochs):
            self.hidden = self.create_hidden_states()

            prediction = self.forward()

            loss = self.loss_function(prediction, self.train_y)
            if epoch % int(self.n_epochs/10) == 0:
                print("Epoch ", epoch, "Error: ", loss.item())
            
            history[epoch] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        plt.plot(history)
    
    def predict(self):
        self.eval()
        prediction = self.forward(train=False)
        return prediction

    def plot_prediction_train(self):
        plt.title('Prediction v Actual - Train')
        plt.plot(self.train_y.cpu().numpy(), color='b')
        plt.plot(self.forward().view(-1).cpu().detach().numpy(), color='r')
        plt.show()
    
    def plot_prediction_train_error(self):
        plt.title('Prediction Error - Train')
        plt.plot(self.forward().view(-1).cpu().detach().numpy() 
                 - self.train_y.cpu().detach().numpy())
        plt.show()
    
    def plot_prediction_test(self):
        plt.title('Prediction v Actuals - Test')
        plt.plot(self.test_y.cpu().detach().numpy(), color='b')
        plt.plot(self.predict().view(-1).cpu().detach().numpy(), color='r')
        plt.show()
    
    def plot_prediction_test_error(self):
        plt.title('Prediction Error - Test')
        plt.plot(self.predict().view(-1).cpu().detach().numpy() 
                 - self.test_y.cpu().detach().numpy())
        plt.show()

