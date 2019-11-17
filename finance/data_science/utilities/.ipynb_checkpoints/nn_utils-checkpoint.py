import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, functional, optim, autograd

from finance.utilities import utils
from finance.data_science.utilities import cluster_utils


class TorchNN(nn.Module):
    def __init__(self,
                 device='cuda:0',
                 train_x=pd.DataFrame,
                 train_y=pd.DataFrame,
                 hidden_size=8,
                 output_size=1,
                 learning_rate=0,
                 momentum=.9):

        super().__init__()

        self.device = device

        self.train_x = torch.tensor(train_x.values
                                   ).to(self.device).float().view(1, -1).detach().requires_grad_(True)
        self.train_y = torch.tensor(train_y.values
                                   ).to(self.device).float()

        self.size = train_x.shape[0] * train_x.shape[1]
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.momentum = momentum

    @property
    def layers(self):
        layers = {
            'linear_one': nn.Linear(in_features=self.size,
                                    out_features=self.hidden_size).float().to(self.device),
            'linear_two': nn.Linear(in_features=self.hidden_size,
                                    out_features=self.output_size).float().to(self.device)
        }
        return layers

    # make this forward pass work for predictions
    def forward(self, data):
        for layer in self.layers:
            data = self.layers.get(layer)(data)
        return data

    @property
    def optimizer(self):
        return optim.SGD([self.train_x], lr=self.learning_rate, momentum=self.momentum)

    @property
    def criterion(self):
        return nn.MSELoss().to(self.device)

    @property
    def n_epochs(self):
        return 2

    def train_network(self):
        running_loss = 0
        optimizer = self.optimizer()
        criterion = self.criterion

        for epoch in range(self.n_epochs):
            for data, target in zip(self.train_x, self.train_y):
                optimizer.zero_grad()

                outputs = self.forward(data).to(self.device)
                loss = criterion(outputs, target).to(self.device)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        return running_loss


class TorchLSTM(nn.Module):
    def __init__(self, 
                 train_x,
                 train_y,
                 test_x,
                 test_y,
                 n_layers=2,
                 hidden_shape=16,
                 output_shape=1,
                 batch_size=1,
                 n_epochs=10,
                 learning_rate=.0001,
                 device='cuda'
                ):
        super(TorchLSTM, self).__init__()
                
        self.n_layers = n_layers
        self.input_shape = train_x.shape[1]
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
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

        self.lstm = nn.LSTM(self.input_shape, self.hidden_shape, self.n_layers).to(self.device)
        self.linear_one = nn.Linear(self.hidden_shape, self.hidden_shape).to(self.device)
        self.linear_two = nn.Linear(self.hidden_shape, self.output_shape).to(self.device)
    
    def reset_network(self):
        self.lstm.reset_parameters()
        self.linear_one.reset_parameters()
        self.linear_two.reset_parameters()
   
    @property
    def loss_function(self):
        return nn.MSELoss(reduction='sum').to(self.device)
    
    @property
    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def create_hidden_states(self):
        # short term memory
        hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_shape).to(self.device)
        
        # long term memory
        cell = torch.zeros(self.n_layers, self.batch_size, self.hidden_shape).to(self.device)
        return hidden, cell

    def forward(self, train=True):
        if train:
            data = self.train_x.view(len(self.train_x), self.batch_size, -1)
        else: 
            data = self.test_x.view(len(self.test_x), self.batch_size, -1)
        
#         output: Tensor[len(self.train_x), 1, self.hidden_size]
#         self.hidden: tuple(Tensor[n_layers, 1, self.hidden_size])
        output, self.hidden = self.lstm(data, self.hidden)
        
        # output: last of len(self.train_x) output
        output = self.linear_one(output)
        output = self.linear_two(output)
        return output
    
    def execute(self):
        optimizer = self.optimizer
        history = np.zeros(self.n_epochs)
        
        self.lstm.weight_hh_l0.data.fill_(0)

        for epoch in range(self.n_epochs):
            self.zero_grad()
            self.hidden = self.create_hidden_states()

            prediction = self.forward()

            loss = self.loss_function(prediction, self.train_y)
            if epoch % int(self.n_epochs/10) == 0:
                print("Epoch ", epoch, "MSE: ", loss.item())
            
            history[epoch] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def predict(self):
        return self.forward(train=False)

    def plot_prediction_train(self):
        plt.title('Prediction v Actual - Train')
        plt.plot(self.train_y.cpu().numpy(), color='b')
        plt.plot(self.forward().view(-1).cpu().detach().numpy(), color='r')
        plt.show()
    
    def plot_prediction_train_error(self):
        plt.title('Prediction Error - Test')
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

            
            
            
            
            
            
            
            
# # Here we define our model as a class
# class LSTM(nn.Module):

#     def __init__(self, 
#                  input_dim, 
#                  hidden_dim, 
#                  batch_size, 
#                  output_dim=1,
#                  num_layers=2):
        
#         super(LSTM, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.batch_size = batch_size
#         self.num_layers = num_layers
#         self.device = 'cuda'

#         # Define the LSTM layer
#         self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers).to(self.device)

#         # Define the output layer
#         self.linear = nn.Linear(self.hidden_dim, output_dim).to(self.device)

#     def init_hidden(self):
#         # This is what we'll initialise our hidden state as
#         return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),
#                 torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))

#     def forward(self, input):
#         # Forward pass through LSTM layer
#         # shape of lstm_out: [input_size, batch_size, hidden_dim]
#         # shape of self.hidden: (a, b), where a and b both 
#         # have shape (num_layers, batch_size, hidden_dim).
#         lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        
#         # Only take the output from the final timetep
#         # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
#         y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
#         return y_pred.view(-1)

# train_x_temp = train_x.drop(cols_to_ignore, axis=1)
# model = LSTM(input_dim=train_x_temp.shape[0]*train_x_temp.shape[1],
#              hidden_dim=4,
#              batch_size=1, 
#              output_dim=1, 
#              num_layers=30)

# x_train = torch.tensor(train_x_temp.values).to('cuda').float().view(1, -1).detach().requires_grad_(True)

# y_train = torch.tensor(train_y.values).to('cuda').float()

# loss_fn = nn.MSELoss(size_average=False).to('cuda')
# optimiser = optim.Adam(model.parameters(), lr=2)

# num_epochs=10
# hist = np.zeros(num_epochs)

# for t in range(num_epochs):
#     # Clear stored gradient
#     model.zero_grad()
    
#     # Initialise hidden state
#     # Don't do this if you want your LSTM to be stateful
#     model.hidden = model.init_hidden()
    
#     # Forward pass
#     y_pred = model(x_train)

#     loss = loss_fn(y_pred, y_train)
#     if t % 1 == 0:
#         print("Epoch ", t, "MSE: ", loss.item())
#     hist[t] = loss.item()

#     # Zero out gradient, else they will accumulate between epochs
#     optimiser.zero_grad()

#     # Backward pass
#     loss.backward()

#     # Update parameters
#     optimiser.step()