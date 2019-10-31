import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster as skcluster
import torch
from torch import nn, functional, optim, autograd

from finance.utilities import utils, cluster_utils


class TorchNN:
    def __init__(self, train_x=pd.DataFrame, train_y=pd.DataFrame,
                 test_x=pd.DataFrame, test_y=pd.DataFrame, learning_rate=0,
                 momentum=.9, hidden_size=None, out_features=1):
        self.train_x = torch.tensor(train_x.values).float().view(1, -1)
        self.train_y = torch.tensor(train_y.values).float()
        # self.test_x = torch.tensor(test_x)
        # self.test_y = torch.tensor(test_y)
        self.size = train_x.shape[0] * train_x.shape[1]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.linear_one = nn.Linear(in_features=self.size, out_features=hidden_size[0]).float()
        self.linear_two = nn.Linear(in_features=hidden_size[0], out_features=out_features).float()

    def forward(self):
        # output = self.linear_one(self.train_x.view(-1, 1))
        output = self.linear_one(self.train_x.view(self.train_x.size(0), -1))
        # output = self.linear_one(self.train_x)
        output = functional.F.relu(output)
        output = self.linear_two(output)
        return output

    def optimizer(self, model):
        optimizer = optim.SGD(nn.Module.parameters(model), lr=self.learning_rate, momentum=self.momentum)
        return optimizer

    def loss_criterion(self):
        return nn.NLLLoss()

    def train_network(self, model):
        for idx, row in self.train_x.iterrows():
            data = autograd.Variable(row)
            target = autograd.Variable(self.train_y[idx])

            self.optimizer.zero_grad()
            net_out = model(data)
            loss = self.loss_criterion(net_out, target)
            loss.backward()

            self.optimizer.step()


if __name__ == '__main__':
    query = """
        select
            e.symbol
            , e.market_datetime
            , e.open
            , e.high
            , e.low
            , e.close
            , e.volume
        from td.equities as e
        where 
        --left(e.symbol, 1) = 'A'
        e.symbol = 'AA'
        order by e.market_datetime
        limit 100
        """
    df = utils.query_db(query=query)
    for col in ['open']:
        df = cluster_utils.normalize(df=df, column=col, subset='symbol')
    df['market_datetime'] = df['market_datetime'].astype(int)

    train = df.copy()
    train = train.drop(columns=['symbol'])
    train['market_datetime'] = train['market_datetime'].astype(int)

    train_target = train['open'].shift(-1)

    split_size = 80000
    train_x = train.iloc[1:split_size]
    train_y = train_target.iloc[1:split_size]

    test_x = train.iloc[split_size + 1:-1]
    test_y = train_target.iloc[split_size + 1:-1]

    model = TorchNN(train_x=train_x, train_y=train_y, hidden_size=(1,)).forward()
    print(model)
