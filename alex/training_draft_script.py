import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import optuna
from tqdm import tqdm
from datetime import datetime
from timeit import default_timer as timer
import joblib

# import custom class and function
from custom import *
from model import RegressionModel
from training import train_step, val_step
from saving import save_best_model
from plotting import plot_prediction_comparison

# create timestamp for saving models
now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

# load and prepare data
X_train_original = pd.read_csv('./training_data/X_train_regression.csv')
X_test = pd.read_csv('./training_data/X_test_regression.csv')
y_train_original = pd.read_csv('./training_data/y_train_regression.csv')
y_test = pd.read_csv('./training_data/y_test_regression.csv')

X_train, X_val, y_train, y_val = train_test_split(X_train_original, y_train_original, test_size=0.2, random_state=42)

# standardise the data
scalerXData = StandardScaler()
X_train_scaled = scalerXData.fit_transform(X_train)
X_val_scaled = scalerXData.transform(X_val)
X_test_scaled = scalerXData.transform(X_test)

scalerYData = StandardScaler()
y_train_scaled = scalerYData.fit_transform(y_train)
y_val_scaled = scalerYData.transform(y_val)
y_test_scaled = scalerYData.transform(y_test)

# setup device agnostic code
try:
    device = setup_cuda_as_device()
    print(f"Using device: {device}")
except CUDANotAvailableError as e:
    raise SystemExit(e)


# Convert data to PyTorch tensors and move to the chosen device
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# create batches
batch_size = 32  # instead of 2**8 for now

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

print(f"Length of train_dataloader: {len(train_dataloader)} batches of {batch_size}")
print(f"Length of val_dataloader: {len(val_dataloader)} batches of {batch_size}")
print(f"Length of test_dataloader: {len(test_dataloader)} batches of {batch_size}")


# creating trial for optuna
def objective(trial):

    # Suggest activation functions for each layer
    activation = trial.suggest_categorical('activation', ['elu', 'sigmoid', 'tanh'])

    # suggest hyperparameters from optuna
    # hidden_size = trial.suggest_int("hidden_size", 2 ** 7, 2 ** 10)
    hidden_size = 500
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout_rate = 0.5

    # print current hyperparameters
    print(f"Trial: {trial.number} | Hyperparameters: lr={lr}, activation={activation}")

    # Create the neural network using the RegressionModel class
    model = RegressionModel(activation,
                            activation,
                            activation,
                            hidden_size=hidden_size,
                            dropout_rate=dropout_rate,
                            input_size=6,
                            output_size=6).to(device)

    criterion = CustomLossFunction().to(device)
    optimiser = torch.optim.Adam(params=model.parameters(), lr=lr)

    patience = 20
    patience_counter = 0
    best_val_loss = np.inf

    epochs = 1000

    # start training loop
    for epoch in tqdm(range(epochs)):
        # Training
        train_step(model=model,
                   data_loader=train_dataloader,
                   criterion=criterion,
                   optimiser=optimiser,
                   device=device)
        # Validation
        val_loss = val_step(model=model,
                            data_loader=val_dataloader,
                            criterion=criterion,
                            device=device)
        # early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f"Stopping early at epoch {epoch + 1}")
            break

    # Testing
    model.load_state_dict(best_weights)
    model.eval()
    with torch.inference_mode():
        y_pred = model(X_test_tensor)
        test_loss = compute_validation_loss(y_pred, y_test_tensor, scalerYData)
    test_loss = test_loss / len(y_test_tensor)

    # save best model
    save_best_model(model, timestamp, str(trial.number))
    print(f"Test loss: {test_loss.item()}")
    return test_loss


folder_name = "models/" + timestamp + "/best_trials/direct"


def save_best_trial(study, trial):
    global model
    global optimiser
    global criterion

    # Continue training only if the current trial is the best one
    if study.best_trial.number == trial.number:
        # Retrieve the best model's state
        os.makedirs(folder_name, exist_ok=True)
        save_path = os.path.join(folder_name, f'best_model_trial_{trial.number}.pth')
        torch.save(model.state_dict(), save_path)  # Save the model weights


start = timer()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5, callbacks=[save_best_trial])

end = timer()
total_time = end - start
print(f"Train time on {device}: {total_time:.3f} seconds")

# Print the best parameters
print(f"Best trial: {study.best_trial.params}")
print(f"Best MSE: {study.best_trial.value}")

completed_trials = study.trials_dataframe()
savePath_4 = os.path.join('./models/' + timestamp + '/completed_trials_regression.csv')
completed_trials.to_csv(savePath_4)

savePath_5 = os.path.join('./models/' + timestamp + '/scalerXData_regression.save')
savePath_6 = os.path.join('./models/' + timestamp + '/scalerYData_regression.save')
joblib.dump(scalerXData, savePath_5)
joblib.dump(scalerYData, savePath_6)

plot_prediction_comparison(model_load_path=os.path.join('./models', timestamp, 'best_trials/direct',
                                                        f'best_model_trial_{study.best_trial.number}.pth'),
                           X_test=X_test_tensor,
                           y_test=y_test_tensor,
                           timestamp=timestamp,
                           labels=['x_absorbent', 'x_co2', 'x_n2', 'y_water', 'y_absorbent', 'y_co2'])
