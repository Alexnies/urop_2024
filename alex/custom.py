import numpy as np
import torch
from torch import nn


class CUDANotAvailableError(Exception):
    pass


def setup_cuda_as_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        raise CUDANotAvailableError("CUDA is not available.")


def compute_validation_loss(y_true, y_pred, scalerYData):
    """
    Compute the RMSE between each element of the predicted values and the true values.

    Args:
        y_true (np.array): True values.
        y_pred (np.array): Predicted values.
        scalerYData (sklearn.preprocessing.StandardScalar()): original StandaredScalar() used on y

    Returns:
        np.array: RMSE for each element.
    """

    # Ensure y_true and y_pred are numpy arrays
    y_true = y_true.cpu().numpy()  # move to CPU and then convert to NumPy array
    y_pred = y_pred.cpu().numpy()

    # Inverse transform y_true and y_pred using the original StandardScaler()
    y_true = scalerYData.inverse_transform(y_true)
    y_pred = scalerYData.inverse_transform(y_pred)

    # Calculate element-wise squared differences
    squared_diffs = (y_true - y_pred) ** 2
    # Compute RMSE for each element
    rmse = np.sqrt(squared_diffs)
    # standardisation (avoid division by 0 by adding epsilon)
    loss = rmse / (y_true + 1e-20)

    return loss.sum().abs()


class CustomLossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, y_true, y_pred):
        mse = self.mse(y_true, y_pred)
        rmse = torch.sqrt(mse + 1e-8)
        # loss = rmse / (y_true + 1e-20)
        return rmse.sum()
