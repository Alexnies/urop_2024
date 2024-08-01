import torch
from torch import nn


class RegressionModel(nn.Module):
    """
    Initialises regression model.

    Args:
        activation[i] (str): (continuous) activation function used between linear layers
        hidden_size (int): Number of hidden units between layers, default 500
        dropout_rate (float): Dropout rate
        input_size (int): Number of input features into the model, default 6
        output_size (int): Number of output features, default 6

    Returns:
        self.layer_stack(x): forward method of the regression model
    """
    def __init__(self,
                 activation_l0,
                 activation_l1,
                 activation_l2,
                 hidden_size_l0: int,
                 hidden_size_l1: int,
                 hidden_size_l2: int,
                 dropout_rate: float = 0.5,
                 input_size: int = 6,
                 output_size: int = 6):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size_l0),
            self.get_activation(activation_l0),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=hidden_size_l0, out_features=hidden_size_l1),
            self.get_activation(activation_l1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=hidden_size_l1, out_features=hidden_size_l2),
            self.get_activation(activation_l2),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=hidden_size_l2, out_features=output_size)
        )

    @staticmethod
    def get_activation(name):
        if name == 'elu':
            return nn.ELU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def forward(self, x):
        return self.layer_stack(x)
