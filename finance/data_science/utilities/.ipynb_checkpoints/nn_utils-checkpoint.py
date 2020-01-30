import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, functional, optim, autograd

from finance.utilities import utils
from finance.data_science.utilities import cluster_utils


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
                 batch_size=None,
                 n_epochs=10,
                 learning_rate=.0001,
                 device='cuda',
                 seed=3,
                 deterministic=True,
                 benchmark=False
                ):
        super(TorchLSTM, self).__init__()
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        
        # Network params
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.input_shape = train_x.shape[1]
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.batch_size = batch_size if batch_size else len(test_x)
        self.train_input = (
            int(len(train_x)/self.batch_size), 
            self.batch_size, 
            self.input_shape
        )
        self.test_input = (
            int(len(test_x)/self.batch_size), 
            self.batch_size, 
            self.input_shape
        )

        # Learning params
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.device = device

        # Data
        self.x_columns = train_x.columns
        self.train_df = None
        self.test_df = None
        
        self.train_x = torch.tensor(
            train_x.values).to(self.device).float().detach().requires_grad_(True).view(self.train_input)
        self.train_y = torch.tensor(
            train_y.values).to(self.device).float()
        
        self.test_x = torch.tensor(
            test_x.values).to(self.device).float().detach().requires_grad_(True).view(self.test_input)
        self.test_y = torch.tensor(
            test_y.values).to(self.device).float()
        
        # Network
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
        # loss = nn.MSELoss(reduction='sum').to(self.device)
        loss = nn.L1Loss(reduction='sum').to(self.device)
        return loss
    
    @property
    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def create_hidden_states(self):
        hidden = torch.zeros(self.n_layers, 
                             self.batch_size, 
                             self.hidden_shape).to(self.device)

        cell = torch.zeros(self.n_layers, 
                           self.batch_size,
                           self.hidden_shape).to(self.device)
        return hidden, cell

    def forward(self, train=True):
        data = self.train_x if train else self.test_x
        output, self.hidden = self.lstm(data, self.hidden)
        output = self.relu(output)
        output = self.linear(output)
        return output
    
    def execute(self):
        optimizer = self.optimizer
        history = np.zeros(self.n_epochs)

        for epoch in range(self.n_epochs):
            self.hidden = self.create_hidden_states()

            prediction = self.forward()

            loss = self.loss_function(prediction.view(-1), self.train_y.view(-1))
            if epoch % int(self.n_epochs/10) == 0:
                print("Epoch ", epoch, "Error: ", loss.item())
            
            history[epoch] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        plt.plot(history)
        self.set_dataframes()
    
    def predict(self):
        self.eval()
        prediction = self.forward(train=False)
        return prediction
    
    def set_dataframes(self):
        args = [[self.train_x, self.train_y, True, self.train_df], 
                [self.test_x, self.test_y, False, self.test_df]]
        for arg in args:
            df = pd.DataFrame(arg[0].view(arg[0].shape[0] * arg[0].shape[1], arg[0].shape[2]
                                         ).cpu().detach().numpy(),
                              columns=self.x_columns)

            actuals = pd.DataFrame(arg[1].view(-1).cpu().detach().numpy(), columns=['actuals'])
            prediction = pd.DataFrame(self.forward(arg[2]).view(-1).cpu().detach().numpy(), 
                                      columns=['prediction'])
            df = df.join(actuals).join(prediction)
            if arg[2]:
                self.train_df = df
            else: self.test_df = df
    
    def plot_prediction_train(self):
        plt.title('Prediction v Actual - Train')
        plt.plot(self.train_df['actuals'], color='b', label='Train')
        plt.plot(self.train_df['prediction'], color='r', label='Prediction')
        plt.legend()
        plt.show()
    
    def plot_prediction_train_error(self):
        plt.title('Prediction Error - Train')
        plt.plot(self.train_df['prediction'] - self.train_df['actuals'])
        plt.show()
    
    def plot_prediction_test(self):
        plt.title('Prediction v Actuals - Test')
        plt.plot(self.test_df['actuals'], color='b', label='Test')
        plt.plot(self.test_df['prediction'], color='r', label='Prediction')
        plt.legend()
        plt.show()
    
    def plot_prediction_test_error(self):
        plt.title('Prediction Error - Test')
        plt.plot(self.test_df['prediction'] - self.test_df['actuals'])
        plt.show()

