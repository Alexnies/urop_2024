import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.model_selection import train_test_split
import joblib
import GPUtil
import optuna
import os
import shutil
from datetime import datetime
import sys

# Some code before the check
print("This code runs before the check.")
should_stop = False
# Get the current date and time
now = datetime.now()

# Format the date and time as a string
timeStamp = now.strftime("%Y_%m_%d_%H_%M_%S")
folder_name_1 = 'models/Training_' + timeStamp + '/weightsDuringTraining'
folder_name_2 = 'models/Training_' + timeStamp + '/bestTrials/direct'
folder_name_3 = 'models/Training_' + timeStamp + '/bestTrials/afterContinuedTraining'
# Define the path where you want to create the folder
path_1 = os.path.join(os.getcwd(), folder_name_1)
path_2 = os.path.join(os.getcwd(), folder_name_2)
path_3 = os.path.join(os.getcwd(), folder_name_3)

# Create the folder
os.makedirs(path_1)
os.makedirs(path_2)
os.makedirs(path_3)

# Check for CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device == "cpu":
    if input('cpu is used as the device, do you wish to continue? y/n') == 'n':
        should_stop = True

# Check the condition
if should_stop:
    print("Stopping the script because the condition is True.")
    sys.exit()

# Set random seed for reproducibility
# torch.manual_seed(42)

def delete_files_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("Folder not found!")
        return

    # Remove each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory and all its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
def compute_validation_loss(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between each element of the
    predicted values and the true values.

    Args:
    y_true (np.array): True values.
    y_pred (np.array): Predicted values.

    Returns:
    np.array: RMSE for each element.
    """
    # Ensure y_true and y_pred are numpy arrays
    y_true = y_true.cpu().numpy()  # Move to CPU and then convert to NumPy array
    y_pred = y_pred.cpu().numpy()

    # what is this??
    y_true = scalerYData.inverse_transform(y_true)
    y_pred = scalerYData.inverse_transform(y_pred)
    # Calculate element-wise squared differences
    squared_diffs = (y_true - y_pred) ** 2

    # Compute RMSE for each element
    rmse = np.sqrt(squared_diffs)

    loss = rmse/(y_true + 1e-20)
    return abs(loss.sum())

def log_scale_dataframe(df, epsilon=1e-20):
    """Applies a log transformation to a DataFrame and returns a NumPy array.

    Args:
        df (pd.DataFrame): The input DataFrame.
        epsilon (float): A small number added to each element to avoid log(0).

    Returns:
        np.ndarray: Log-scaled data as a NumPy array.
    """
    log_scaled_data = np.log(df + epsilon)
    return log_scaled_data.to_numpy()


def inverse_log_scale_dataframe(log_scaled_array):
    """Applies the exponential function to a log-scaled NumPy array and returns a NumPy array.

    Args:
        log_scaled_array (np.ndarray): The log-scaled data as a NumPy array.

    Returns:
        np.ndarray: Data reverted from log scale to its original scale.
    """
    original_data = np.exp(log_scaled_array)
    return original_data

class CustomLossFunction(nn.Module):
    def __init__(self):
        super(CustomLossFunction, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_true, y_pred):
        # Calculate MSE
        mse = self.mse(y_true, y_pred)
        # Calculate RMSE
        rmse = torch.sqrt(mse + 1e-8)

        loss = rmse / (y_true + 1e-20)
        # Calculate custom loss
        # Reduce the loss to a single scalar
        return rmse.sum()

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.5, input_size=10, output_size=6):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size),
        nn.ELU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(hidden_size, hidden_size),
        nn.ELU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(hidden_size, hidden_size),
        nn.ELU(),
        nn.Dropout(p=dropout_rate),
        # nn.Linear(hidden_size, hidden_size),
        # nn.ReLU(),
        # nn.Dropout(p=dropout_rate),
        nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.layers(x)


# Load and prepare data
X_train = pd.read_csv('./trainingData/X_train_regression.csv')
X_test = pd.read_csv('./trainingData/X_test_regression.csv')
y_train = pd.read_csv('./trainingData/y_train_regression.csv')
y_test = pd.read_csv('./trainingData/y_test_regression.csv')
# Assuming X_train and y_train are your full training data and labels
X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# Standardize the data
scalerXData = StandardScaler()
X_train_scaled = scalerXData.fit_transform(X_train_new)
X_val_scaled = scalerXData.transform(X_val)
X_test_scaled = scalerXData.transform(X_test)


scalerYData = StandardScaler()
y_train_scaled = scalerYData.fit_transform(y_train_new)
y_val_scaled = scalerYData.transform(y_val)
y_test_scaled = scalerYData.transform(y_test)

# scalerYData = StandardScaler()
# y_train_scaled = log_scale_dataframe(y_train_new)
# y_val_scaled = log_scale_dataframe(y_val)
# y_test_scaled = log_scale_dataframe(y_test)


# Convert data to PyTorch tensors and move to the chosen device
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 2**8
print("Batch size is ", batch_size)
print("Which is ", batch_size*100/len(train_dataset), "% of the training set")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # No need to shuffle test data
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation data

# Define hyperparameters
num_epochs = 1000

def save_best_trial(study, trial):
    global model
    global optimizer
    global criterion
    # Continue training only if the current trial is the best one
    if study.best_trial.number == trial.number:
        # Retrieve the best model's state
        savePath_2 = os.path.join(folder_name_2, f'best_model_trial_{trial.number}.pth')
        torch.save(model.state_dict(),savePath_2)  # Save the model weights

def continue_training_callback(study, trial):
    global model
    global optimizer
    global criterion
    # Continue training only if the current trial is the best one
    if study.best_trial.number == trial.number:
        # Retrieve the best model's state
        savePath_1 = os.path.join(folder_name_1, f'best_model_trial_{trial.number}.pth')
        model.load_state_dict(torch.load(savePath_1))
        model.to(device)
        savePath_2 = os.path.join(folder_name_2, f'best_model_trial_{trial.number}.pth')
        torch.save(model.state_dict(),savePath_2)  # Save the model weights
        # Set extended training parameters
        extended_epochs = 500  # Number of extra epochs to train
        patience = 50
        best_val_loss = np.inf
        patience_counter = 0

        # Extended training loop
        for epoch in range(extended_epochs):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Average the loss
            # epoch_loss /= len(train_loader)
            average_training_loss = epoch_loss / len(train_loader.dataset)
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    val_outputs = model(X_val)
                    batch_loss = criterion(val_outputs, y_val)
                    val_loss += batch_loss.item()

            # val_loss /= len(val_loader)
            average_val_loss = val_loss / len(val_loader.dataset)
            print(
                f'Extended Training - Epoch {epoch + 1}, Training Loss: {average_training_loss:.4f}, Validation Loss: {average_val_loss:.4f}')

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter > patience:
                print("Stopping extended training early due to no improvement.")
                break
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            test_loss = compute_validation_loss(y_pred, y_test_tensor)
            test_loss = test_loss / len(y_test_tensor)
            savePath_3 = os.path.join(folder_name_3, f'best_model_trial_{trial.number}.pth')
            torch.save(model.state_dict(), savePath_3)
        print(f'Test Loss: {test_loss.item()}')
        return test_loss
def objective(trial):
    global model
    global optimizer
    global criterion
    hidden_size = trial.suggest_int('hidden_size', 2**7, 2**10)
    dropout_rate = 0.5#trial.suggest_float('dropout_rate', 0.3, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # Print current hyperparameters
    print(f"Trial {trial.number}, Hyperparameters: lr={lr}, n_units={hidden_size}, dropout_rate={dropout_rate}")
    model = NeuralNetwork(hidden_size=hidden_size, dropout_rate=dropout_rate, input_size=10, output_size=6)
    model.to(device)
    criterion = CustomLossFunction().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    patience = 20
    best_val_loss = np.inf
    patience_counter = 0


    for epoch in range(num_epochs):

        epoch_loss = 0
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(y_batch,outputs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        #epoch_loss /= len(train_loader)  # Average the loss over all training batches
        average_training_loss = epoch_loss / len(train_loader.dataset)
        # Start the Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                batch_loss = criterion(y_val,val_outputs)
                val_loss += batch_loss.item()

        #val_loss /= len(val_loader)  # Average the loss over all validation batches
        average_val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch {epoch + 1}, Training Loss: {average_training_loss:.4f}, Validation Loss: {average_val_loss:.4f}')  # Update with actual training loss
        #GPUs = GPUtil.getGPUs()
        # for gpu in GPUs:
        #     print(f"GPU: {gpu.id}, load: {gpu.load * 100}%, GPU temp: {gpu.temperature} C")
        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f"Stopping early at epoch {epoch + 1}")
            break

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        test_loss = compute_validation_loss(y_pred, y_test_tensor)
    test_loss = test_loss / len(y_test_tensor)
    savePath_1 = os.path.join(folder_name_1, f'best_model_trial_{trial.number}.pth')
    torch.save(model.state_dict(), savePath_1)  # Save the model weights
    print(f'Test Loss: {test_loss.item()}')
    return test_loss

start_time = time.time()  # Capture the end time after the operation completes


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500, callbacks=[save_best_trial])


end_time = time.time()  # Capture the end time after the operation completes
print(f'Training took {end_time - start_time} seconds')

# Print the best parameters
print(f"Best trial: {study.best_trial.params}")
print(f"Best MSE: {study.best_trial.value}")

completed_trials = study.trials_dataframe()
savePath_4 = os.path.join('./models/Training_' + timeStamp + '/completed_trials_regression.csv')
completed_trials.to_csv(savePath_4)

savePath_5 = os.path.join('./models/Training_' + timeStamp + '/scalerXData_regression.save')
savePath_6 = os.path.join('./models/Training_' + timeStamp + '/scalerYData_regression.save')
joblib.dump(scalerXData, savePath_5)
joblib.dump(scalerYData, savePath_6)
