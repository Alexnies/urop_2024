import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import optuna
from datetime import datetime
import matplotlib.pyplot as plt

# import custom functions
from custom import CUDANotAvailableError, setup_cuda_as_device
from model import RegressionModel


# define GLOBAL variables
BATCH_SIZE = 32
DIR = os.getcwd()
EPOCHS = 1000
TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
try:
    DEVICE = setup_cuda_as_device()  # setup device agnostic code
    print(f"Using device: {DEVICE}")
except CUDANotAvailableError as e:
    raise SystemExit(e)


def save_best_trial(study, trial):
    global model
    global MODEL_SAVE_PATH  # for plotting purposes

    if study.best_trial.number == trial.number:
        folder_path = os.path.join(DIR, "experiments", TIMESTAMP, "models")
        os.makedirs(folder_path, exist_ok=True)
        MODEL_SAVE_PATH = os.path.join(folder_path, f'best_model_trial_{trial.number}.pth')
        torch.save(model.state_dict(), MODEL_SAVE_PATH)


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


def define_model(trial):
    """
        Currently the model is optimising the i) hidden size and ii) activation function
    """
    n_layers = trial.suggest_int("layers", 1, 5)
    # n_layers = 3  # can be changed later
    layers = []

    in_features = 6
    for i in range(n_layers):
        # hidden_size = trial.suggest_int("n_units_l{}".format(i), 2 ** 7, 2 ** 10)
        hidden_size = 500
        layers.append(nn.Linear(in_features, hidden_size))
        # activation = trial.suggest_categorical(f'activation_l{i}', ['elu', 'sigmoid', 'tanh'])
        activation = 'elu'
        layers.append(get_activation(activation))
        p = 0.5  # can be changed later
        layers.append(nn.Dropout(p))

        in_features = hidden_size

    layers.append(nn.Linear(in_features, 6))

    return nn.Sequential(*layers)


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


def objective(trial):
    global model  # for save_best_trial function
    # Generate the model
    model = define_model(trial).to(DEVICE)

    # Generate the optimisers
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimiser = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = RMSELoss()

    # get the data loaders
    train_loader, valid_loader, test_loader = get_data_loader()

    # print current hyperparameters
    print(f"Trial: {trial.number} | Hyperparameters: lr: {lr}")

    # start training loop
    for epoch in tqdm(range(EPOCHS)):
        # Training
        train_loss = 0

        # put model into training mode
        model.train()

        # add a loop to loop through the training batches
        for batch, (X, y) in enumerate(train_loader):
            # put data on target device
            X, y = X.to(DEVICE), y.to(DEVICE)

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

        # divide total train loss by length of train dataloader
        train_loss /= len(train_loader)

        # Validation
        val_loss = 0

        # put the model in eval mode
        model.eval()
        with torch.inference_mode():
            for X, y in valid_loader:
                # send the data to the target device
                X, y = X.to(DEVICE), y.to(DEVICE)
                # 1. forward pass
                val_pred = model(X)
                # 2. calculate the loss
                val_loss += criterion(val_pred, y)

            # adjust metrics and print out
            val_loss /= len(valid_loader)

        # Testing
        test_loss = 0

        # put the model in eval mode
        model.eval()
        with torch.inference_mode():
            for X, y in test_loader:
                # send the data to the target device
                X, y = X.to(DEVICE), y.to(DEVICE)
                # 1. forward pass
                test_pred = model(X)
                # 2. calculate the loss
                test_loss += criterion(test_pred, y)

            # adjust metrics and print out
            test_loss /= len(test_loader)

        trial.report(test_loss, epoch)

        # Handle pruning based on intermediate value?

    print(f"Train loss: {train_loss:.5f} | Validation loss: {val_loss:.5f} | Test loss: {test_loss:.5f}")

    return test_loss


def rename_state_dict_keys(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = 'layer_stack.' + key
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def plot_prediction_comparison(best_trial_num: int,
                               trials_df: pd.DataFrame,
                               labels: list,
                               show: bool = True):
    # Load the parameters of the best model
    activation_l0 = trials_df.at[best_trial_num, "params_activation_l0"]
    activation_l1 = trials_df.at[best_trial_num, "params_activation_l1"]
    activation_l2 = trials_df.at[best_trial_num, "params_activation_l2"]
    hidden_units_l0 = trials_df.at[best_trial_num, "params_n_units_l0"]
    hidden_units_l1 = trials_df.at[best_trial_num, "params_n_units_l1"]
    hidden_units_l2 = trials_df.at[best_trial_num, "params_n_units_l2"]

    # Load the model
    model = RegressionModel(activation_l0=activation_l0,
                            activation_l1=activation_l1,
                            activation_l2=activation_l2,
                            hidden_size_l0=hidden_units_l0,
                            hidden_size_l1=hidden_units_l1,
                            hidden_size_l2=hidden_units_l2)
    old_state_dict = torch.load(MODEL_SAVE_PATH)
    new_state_dict = rename_state_dict_keys(old_state_dict)
    model.load_state_dict(new_state_dict)
    model.to("cpu")
    model.eval()

    X_test = X_test_tensor.to("cpu")
    y_test = y_test_tensor.to("cpu")

    with torch.inference_mode():
        y_pred = model(X_test)

    # inverse scale the dataset
    # y_test = y_data_scalar.inverse_transform(y_test)
    # y_pred = y_data_scalar.inverse_transform(y_pred)

    # turn into numpy arrays
    y_test = y_test.numpy()
    y_pred = y_pred.numpy()

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
        figures_dir = os.path.join(os.getcwd(), "experiments", TIMESTAMP, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        fig_name = f"{labels[i]}.png"
        save_path = os.path.join(figures_dir, fig_name)
        plt.savefig(save_path, bbox_inches='tight')

        # show figure
        if show:
            plt.show()

        # close figure
        plt.close()


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, callbacks=[save_best_trial])

print("Study statistics: ")
print(f"   Number of finished trials: {len(study.trials)}")

print(f"Best trial:")
trial = study.best_trial

print(f"   Value: {trial.value}")

print("   Params: ")
for key, value in trial.params.items():
    print(f"   {key}: {value}")

completed_trials = study.trials_dataframe()
save_path = os.path.join(DIR, "experiments", TIMESTAMP, 'completed_trials.csv')
completed_trials.to_csv(save_path)

fig = optuna.visualization.plot_param_importances(study)
fig.show()

plot_prediction_comparison(best_trial_num=study.best_trial.number,
                           trials_df=completed_trials,
                           labels=['x_absorbent', 'x_co2', 'x_n2', 'y_water', 'y_absorbent', 'y_co2'],
                           show=True)
