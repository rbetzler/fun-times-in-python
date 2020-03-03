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
                 hidden_shape=100,
                 output_shape=1,
                 batch_size=None,
                 n_training_batches=1,
                 n_epochs=100,
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
        self.input_shape = train_x.shape[1]
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.batch_size = batch_size if batch_size else len(test_x)
        self.n_training_batches = n_training_batches
        self.train_input = (
            int(len(train_x)/self.batch_size/self.n_training_batches),
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
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        
        # Network
        self.lstm = nn.LSTM(input_size=self.input_shape, 
                            hidden_size=self.hidden_shape,
                            num_layers=n_layers,
                            bias=bias,
                            dropout=dropout).to(self.device)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.hidden_shape, self.output_shape).to(self.device)
        
        self.lstm = nn.DataParallel(self.lstm)
        self.relu = nn.DataParallel(self.relu)
        self.linear = nn.DataParallel(self.linear)
    
    def reset_network(self):
        self.lstm.reset_parameters()
        self.linear.reset_parameters()

    def training_data(self):
        x = np.array_split(self.train_x, self.n_training_batches)
        y = np.array_split(self.train_y, self.n_training_batches)
        
        data = []
        for n in range(0, self.n_training_batches):
            data.append((x[n], y[n]))
        return data

    @property
    def loss_function(self):
        loss = nn.L1Loss(reduction='sum').to(self.device)
        return loss
    
    @property
    def optimizer(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

#     def create_hidden_states(self):
#         hidden = torch.zeros(self.n_layers, 
#                              self.batch_size, 
#                              self.hidden_shape).to(self.device)

#         cell = torch.zeros(self.n_layers, 
#                            self.batch_size,
#                            self.hidden_shape).to(self.device)
#         return hidden, cell

    def forward(self, data):
#         TODO: figure out what these hidden states actually do
#         output, self.hidden = self.lstm(data, self.hidden)

        self.lstm.module.flatten_parameters()
        output, _ = self.lstm(data, None)
        output = self.relu(output)
        output = self.linear(output)
        return output

    def fit(self):
        optimizer = self.optimizer
        history = []

        batch = 0
        for data in self.training_data():
            batch += 1
            x = torch.tensor(data[0].values).to(self.device).float().detach().requires_grad_(True).view(self.train_input)
            y = torch.tensor(data[1].values).to(self.device).float()

            for epoch in range(self.n_epochs):
                prediction = self.forward(x)

                loss = self.loss_function(prediction.view(-1), y.view(-1))
                if epoch % int(self.n_epochs/10) == 0:
                    print(f'Batch {batch}, Epoch {epoch}, Loss {loss.item()}')

                history.append([batch, epoch, loss.item()])

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
        df_history = pd.DataFrame(history, columns=['batch', 'epoch', 'loss'])
        
        plt.title('Cumulative Loss by Epoch')
        plt.plot(df_history.groupby('epoch').sum()['loss'])

    def predict(self):
        self.eval()
        x = torch.tensor(self.test_x.values).to(self.device).float().detach().view(self.test_input)
        prediction = self.forward(x)
        return prediction

    @property
    def prediction_df(self):
        df = self.test_x.copy()
        df['prediction'] = self.predict().view(-1).cpu().detach().numpy()
        return df

    # TODO:
    #   Add and refined basic performance features