import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# import custom functions
from custom import CUDANotAvailableError, setup_cuda_as_device
from model import RegressionModel

# define GLOBAL variables
BATCH_SIZE = 32
DIR = os.getcwd()
EPOCHS = 100
TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
MODEL_SAVE_PATH = '/home/an1821/Desktop/urop_2024/alex/experiments/2024_07_25_15_00_23/models/best_model_trial_68.pth'
try:
    DEVICE = setup_cuda_as_device()  # setup device agnostic code
    print(f"Using device: {DEVICE}")
except CUDANotAvailableError as e:
    raise SystemExit(e)


def get_data_loader():
    global X_test_tensor
    global y_test_tensor
    global X_data_scalar
    global y_data_scalar  # for plotting purposes

    # load and prepare data
    X_train_original = pd.read_csv('./training_data/X_train_regression.csv')
    X_test = pd.read_csv('./training_data/X_test_regression.csv')
    y_train_original = pd.read_csv('./training_data/y_train_regression.csv')
    y_test = pd.read_csv('./training_data/y_test_regression.csv')

    X_train, X_val, y_train, y_val = train_test_split(X_train_original, y_train_original,
                                                      test_size=0.2,
                                                      random_state=42)

    # standardise the data
    X_data_scalar = StandardScaler()
    X_train_scaled = X_data_scalar.fit_transform(X_train)
    X_val_scaled = X_data_scalar.transform(X_val)
    X_test_scaled = X_data_scalar.transform(X_test)

    y_data_scalar = StandardScaler()
    y_train_scaled = y_data_scalar.fit_transform(y_train)
    y_val_scaled = y_data_scalar.transform(y_val)
    y_test_scaled = y_data_scalar.transform(y_test)

    # Convert data to PyTorch tensors and move to the chosen device
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(DEVICE)

    # create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    valid_loader = DataLoader(dataset=val_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    print(f"\nLength of train_dataloader: {len(train_loader)} batches of {BATCH_SIZE}")
    print(f"Length of val_dataloader: {len(valid_loader)} batches of {BATCH_SIZE}")
    print(f"Length of test_dataloader: {len(test_loader)} batches of {BATCH_SIZE}\n")

    return train_loader, valid_loader, test_loader


def get_activation(name):
    """
        Function used to select the activation function from nn
    """
    if name == 'elu':
        return nn.ELU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation function: {name}")


def rename_state_dict_keys(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = 'layer_stack.' + key
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def define_model():
    """

    """

    # Load the model
    model = RegressionModel(activation_l0='elu',
                            activation_l1='elu',
                            activation_l2='elu',
                            hidden_size_l0=500,
                            hidden_size_l1=500,
                            hidden_size_l2=500,
                            dropout_rate=0)
    #old_state_dict = torch.load(MODEL_SAVE_PATH)
    #new_state_dict = rename_state_dict_keys(old_state_dict)
    #model.load_state_dict(new_state_dict)

    return model


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_true, y_pred):
        # Calculate MSE
        mse = self.mse(y_true, y_pred)
        # Calculate RMSE
        rmse = torch.sqrt(mse + 1e-8)
        # calculate loss
        loss = rmse.mean()  # worth experimenting later
        # loss = rmse / (y_true + 1e-20)  # relative loss only worked without scaling
        return loss


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               optimiser: torch.optim.Optimizer,
               device: torch.device):
    """
    Performs a training with model trying to learn on data_loader.
    """
    train_loss = 0

    # put model into training mode
    model.to(device)
    model.train()

    # add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):
        # put data on target device
        X, y = X.to(device), y.to(device)

        # 1. forward pass
        y_pred = model(X)
        # 2. calculate loss (per batch)
        loss = criterion(y_pred, y)
        train_loss += loss  # accumulate train loss
        # 3. optimiser zero grad
        optimiser.zero_grad()
        # 4. loss backward
        loss.backward()
        # 5. optimser step (update the model's parameters once per batch)
        optimiser.step()

        # Print out what's happening
        # if batch % 200 == 0:
            # print(f"Looked at {batch * len(X)}/{len(data_loader.dataset)} samples.")

    # divide total train loss by length of train dataloader
    train_loss /= len(data_loader)
    # print(f"Train loss: {train_loss:.5f}")
    return train_loss


def val_step(model: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module,
             device: torch.device):
    """
    Performs a testing loop step on model going over data_loader.
    """
    val_loss = 0

    # put the model in eval mode
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # send the data to the target device
            X, y = X.to(device), y.to(device)
            # 1. forward pass
            val_pred = model(X)
            # 2. calculate the loss
            val_loss += criterion(val_pred, y)

        # adjust metrics and print out
        val_loss /= len(data_loader)
        # print(f"Validation loss: {val_loss:.5f}")
    return val_loss


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              device: torch.device):
    """
    Performs a testing loop step on model going over data_loader.
    """
    test_loss = 0

    # put the model in eval mode
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # send the data to the target device
            X, y = X.to(device), y.to(device)
            # 1. forward pass
            test_pred = model(X)
            # 2. calculate the loss
            test_loss += criterion(test_pred, y)

        # adjust metrics and print out
        test_loss /= len(data_loader)
        # print(f"Test loss: {test_loss:.5f}")

    return test_loss


def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               device: torch.device):
    """
    Return a dictionary containing the results of model predicting on data_loader.
    """
    loss = 0

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # accumulate the loss and acc values per batch
            loss += criterion(y_pred, y)

        # scale loss to find the average loss per batch
        loss /= len(data_loader)

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": loss.item()}


def plot_prediction_comparison(model,
                               labels: list,
                               show: bool = True):
    model.to("cpu")
    model.eval()

    X_test = X_test_tensor.to("cpu")
    y_test = y_test_tensor.to("cpu")

    with torch.inference_mode():
        y_pred = model(X_test)

    # inverse scale the dataset
    y_test = y_data_scalar.inverse_transform(y_test)
    y_pred = y_data_scalar.inverse_transform(y_pred)

    # Plot predictions vs actual values for each feature
    num_features = y_test.shape[1]
    for i in range(num_features):
        plt.figure()
        plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)

        # Plot the 45-degree line
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Comparison for {labels[i]}')

        # save figure
        # figures_dir = os.path.join(os.getcwd(), "experiments", TIMESTAMP, 'figures')
        # os.makedirs(figures_dir, exist_ok=True)
        # fig_name = f"{labels[i]}.png"
        # save_path = os.path.join(figures_dir, fig_name)
        # plt.savefig(save_path, bbox_inches='tight')

        # show figure
        if show:
            plt.show()

        # close figure
        plt.close()


# get the data loaders
train_loader, valid_loader, test_loader = get_data_loader()
model = define_model()
optimiser = torch.optim.Adam(params=model.parameters(), lr=0.00018813727196373314)  # manual for now
criterion = RMSELoss()
# create lists
epoch_arr = []
train_loss_arr = []
valid_loss_arr = []
test_loss_arr = []
# create loop
for epoch in tqdm(range(EPOCHS)):
    epoch_arr.append(epoch)
    train_loss = train_step(model,
                            train_loader,
                            criterion,
                            optimiser,
                            DEVICE)
    train_loss_arr.append(train_loss.item())
    valid_loss = val_step(model,
                          valid_loader,
                          criterion,
                          DEVICE)
    valid_loss_arr.append(valid_loss.item())
    test_loss = test_step(model,
                          test_loader,
                          criterion,
                          DEVICE)
    test_loss_arr.append(test_loss.item())

plt.figure()
plt.plot(epoch_arr, train_loss_arr, c='r', label='train loss')
plt.plot(epoch_arr, valid_loss_arr, c='b', label='validation loss')
plt.plot(epoch_arr, test_loss_arr, c='g', label='test loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.close()

plot_prediction_comparison(model,
                           labels=['x_absorbent', 'x_co2', 'x_n2', 'y_water', 'y_absorbent', 'y_co2'])
