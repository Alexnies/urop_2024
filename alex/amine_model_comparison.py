import os

# must create these environment variabels to run gOPython
os.environ['GPROMSHOME'] = '/usr/local/pse/gPROCESS_2023/gPROMS-core_2022.2.2.55277'
os.environ['PSELMD_LICENSE_FILE'] = '27006@gproms.cc.ic.ac.uk'

# must import gopython immediately after creating the temp env vars
import gopython
import io
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd

pd.set_option("display.max_rows", 5)
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import warnings
from tqdm.auto import tqdm

# Define global variables
TIMESTAMP = "2024_07_25_15_00_23"


def get_hidden_size(trial_number, csv_file_path):
    df = pd.read_csv(csv_file_path)
    row = df[df['number'] == trial_number]
    if not row.empty:
        return int(row['params_hidden_size'].values[0])
    return None


class CustomLossFunction(nn.Module):
    def __init__(self):
        super(CustomLossFunction, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_true, y_pred):
        return torch.sqrt(self.mse(y_true, y_pred))


class NeuralNetworkRegression(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.5, input_size=10, output_size=6):
        super(NeuralNetworkRegression, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.ELU(),
                                    nn.Dropout(p=dropout_rate),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ELU(),
                                    nn.Dropout(p=dropout_rate),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ELU(),
                                    # nn.Dropout(p=dropout_rate),
                                    # nn.Linear(hidden_size, hidden_size),
                                    # nn.ReLU(),
                                    nn.Dropout(p=dropout_rate),
                                    nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.layers(x)


class NeuralNetworkClassification(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.5, input_size=6, output_size=1):
        super(NeuralNetworkClassification, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def classify_and_regress(temperature, pressure, composition):
    global scalerXDataRegression
    global scalerClassification
    global device
    global model_classification
    global model_regression

    input_data = np.concatenate([
        np.array([temperature]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([pressure]).reshape(1, 1),  # Reshape to (1, 1)
        np.array(composition).reshape(1, -1)
        # Reshape composition to have 1 row and -1 lets numpy decide the number of columns
    ], axis=1)
    # input_data_reshaped = input_data.reshape(1, -1)
    # Assuming scalerClassification and scalerRegression do not need to be moved to device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        input_data_scaled_classification = scalerClassification.transform(input_data)
        input_data_scaled_regression = scalerXDataRegression.transform(input_data)

    # Convert scaled data to tensors and move them to the specified device
    input_tensor_classification = torch.FloatTensor(input_data_scaled_classification).to(device)
    input_tensor_regression = torch.FloatTensor(input_data_scaled_regression).to(device)

    # Move models to the device
    model_classification.to(device)
    model_regression.to(device)

    # Perform classification
    with torch.no_grad():
        classification_result = model_classification(input_tensor_classification)

    with torch.no_grad():
        regression_result = model_regression(input_tensor_regression)

    # Convert the tensors to NumPy arrays
    regression_result = regression_result.cpu().numpy()
    classification_result = classification_result.cpu().numpy()

    # Convert the NumPy arrays to pandas DataFrames
    regression_result = pd.DataFrame(regression_result,
                                     columns=["x_ABSORBENT", "x_CO2", "x_N2", "y_water", "y_ABSORBENT", "y_CO2"])

    # Use the .clip() method to set all values below 0 to 0 in both DataFrames
    regression_result = regression_result.clip(lower=0)

    regression_result["x_water"] = 1 - (
            regression_result["x_ABSORBENT"] + regression_result["x_CO2"] +
            regression_result["x_N2"])
    regression_result["y_N2"] = 1 - (
                regression_result["y_water"] + regression_result["y_ABSORBENT"] + regression_result["y_CO2"])

    x_values = ["x_water", "x_ABSORBENT", "x_CO2", "x_N2"]
    y_values = ["y_water", "y_ABSORBENT", "y_CO2", "y_N2"]

    x = 0 + (1 - classification_result) * regression_result[x_values]
    y = composition * classification_result + (1 - classification_result) * regression_result[y_values]
    output = pd.concat([x, y], axis=1)
    return 1, output.values.tolist()[0]


def regress(temperature, pressure, z_H2O, z_ABSORBENT, z_CO2, z_N2, mw_H2O, mw_ABSORBENT, mw_CO2, mw_N2):
    global scalerXDataRegression
    global scalerYDataRegression
    global device
    global model_regression

    input_data = np.concatenate([
        np.array([temperature]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([pressure]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_H2O]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_ABSORBENT]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_CO2]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_N2]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([mw_H2O]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([mw_ABSORBENT]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([mw_CO2]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([mw_N2]).reshape(1, 1)  # Reshape to (1, 1)
        # Reshape composition to have 1 row and -1 lets numpy decide the number of columns
    ], axis=1)
    # input_data_reshaped = input_data.reshape(1, -1)
    # Assuming scalerClassification and scalerRegression do not need to be moved to device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        input_data_scaled_regression = scalerXDataRegression.transform(input_data)

    # Convert scaled data to tensors and move them to the specified device
    input_tensor_regression = torch.FloatTensor(input_data_scaled_regression).to(device)

    # Move models to the device
    model_regression.to(device)

    with torch.no_grad():
        regression_result = model_regression(input_tensor_regression)

    # Convert the tensors to NumPy arrays

    regression_result = scalerYDataRegression.inverse_transform(regression_result.cpu().numpy())

    # Convert the NumPy arrays to pandas DataFrames
    regression_result = pd.DataFrame(regression_result,
                                     columns=["x_ABSORBENT", "x_CO2", "x_N2", "y_water", "y_ABSORBENT", "y_CO2"])

    # Use the .clip() method to set all values below 0 to 0 in both DataFrames
    regression_result = regression_result.clip(lower=0)

    x_values = ["x_ABSORBENT", "x_CO2", "x_N2"]
    y_values = ["y_water", "y_ABSORBENT", "y_CO2"]

    output = pd.concat([regression_result[x_values], regression_result[y_values]], axis=1)

    return 1, output.values.tolist()[0]


def deriv_temperature(temperature, pressure, z_H2O, z_ABSORBENT, z_CO2, z_N2):
    global scalerXDataRegression
    global scalerYDataRegression
    global device
    global model_regression

    input_data = np.concatenate([
        np.array([temperature]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([pressure]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_H2O]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_ABSORBENT]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_CO2]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_N2]).reshape(1, 1)  # Reshape to (1, 1)
        # Reshape composition to have 1 row and -1 lets numpy decide the number of columns
    ], axis=1)
    # input_data_reshaped = input_data.reshape(1, -1)
    # Assuming scalerClassification and scalerRegression do not need to be moved to device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        input_data_scaled_regression = scalerXDataRegression.transform(input_data)

    # Convert scaled data to tensor, enable gradient, and move to device
    input_tensor_regression = torch.tensor(input_data_scaled_regression, dtype=torch.float32, device=device)
    input_tensor_regression.requires_grad = True

    # Get scaling factor for output and convert to tensor
    sigma_x = torch.tensor(scalerXDataRegression.scale_, dtype=torch.float32, device=device)
    sigma_y = torch.tensor(scalerYDataRegression.scale_, dtype=torch.float32, device=device)
    # Move models to the device
    model_regression.to(device)
    # Perform forward pass
    regression_result = model_regression(input_tensor_regression)

    # Compute gradients for each output w.r.t. temperature (first input)
    temperature_grads = []
    for i in range(regression_result.shape[1]):
        model_regression.zero_grad()
        regression_result[:, i].backward(retain_graph=True)
        # Collect the gradient for temperature only (index 0)
        temperature_grads.append(input_tensor_regression.grad[:, 0].clone())
        input_tensor_regression.grad.zero_()  # Reset gradients for the next output

    # Adjust gradients by multiplying with sigma

    adjusted_temperature_grads = [g * (sigma_y[i] / (sigma_x[0])) for i, g in enumerate(temperature_grads)]

    # Return numpy array of adjusted gradients for temperature
    return 1, [grad.item() for grad in adjusted_temperature_grads]


def deriv_pressure(temperature, pressure, z_H2O, z_ABSORBENT, z_CO2, z_N2):
    global scalerXDataRegression
    global scalerYDataRegression
    global device
    global model_regression

    input_data = np.concatenate([
        np.array([temperature]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([pressure]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_H2O]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_ABSORBENT]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_CO2]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_N2]).reshape(1, 1)  # Reshape to (1, 1)
        # Reshape composition to have 1 row and -1 lets numpy decide the number of columns
    ], axis=1)
    # input_data_reshaped = input_data.reshape(1, -1)
    # Assuming scalerClassification and scalerRegression do not need to be moved to device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        input_data_scaled_regression = scalerXDataRegression.transform(input_data)

    # Convert scaled data to tensor, enable gradient, and move to device
    input_tensor_regression = torch.tensor(input_data_scaled_regression, dtype=torch.float32, device=device)
    input_tensor_regression.requires_grad = True

    # Get scaling factor for output and convert to tensor
    sigma_x = torch.tensor(scalerXDataRegression.scale_, dtype=torch.float32, device=device)
    sigma_y = torch.tensor(scalerYDataRegression.scale_, dtype=torch.float32, device=device)
    # Move models to the device
    model_regression.to(device)
    # Perform forward pass
    regression_result = model_regression(input_tensor_regression)

    # Compute gradients for each output w.r.t. pressure (second input)
    pressure_grads = []
    for i in range(regression_result.shape[1]):
        model_regression.zero_grad()
        regression_result[:, i].backward(retain_graph=True)
        # Collect the gradient for temperature only (index 0)
        pressure_grads.append(input_tensor_regression.grad[:, 1].clone())
        input_tensor_regression.grad.zero_()  # Reset gradients for the next output

    # Adjust gradients by multiplying with sigma
    adjusted_pressure_grads = [g * (sigma_y[i] / sigma_x[1]) for i, g in enumerate(pressure_grads)]

    # Return numpy array of adjusted gradients for pressure
    return 1, [grad.item() for grad in adjusted_pressure_grads]


def deriv_z_H2O(temperature, pressure, z_H2O, z_ABSORBENT, z_CO2, z_N2):
    global scalerXDataRegression
    global scalerYDataRegression
    global device
    global model_regression

    input_data = np.concatenate([
        np.array([temperature]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([pressure]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_H2O]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_ABSORBENT]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_CO2]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_N2]).reshape(1, 1)  # Reshape to (1, 1)
        # Reshape composition to have 1 row and -1 lets numpy decide the number of columns
    ], axis=1)
    # input_data_reshaped = input_data.reshape(1, -1)
    # Assuming scalerClassification and scalerRegression do not need to be moved to device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        input_data_scaled_regression = scalerXDataRegression.transform(input_data)

    # Convert scaled data to tensor, enable gradient, and move to device
    input_tensor_regression = torch.tensor(input_data_scaled_regression, dtype=torch.float32, device=device)
    input_tensor_regression.requires_grad = True

    # Get scaling factor for output and convert to tensor
    sigma_x = torch.tensor(scalerXDataRegression.scale_, dtype=torch.float32, device=device)
    sigma_y = torch.tensor(scalerYDataRegression.scale_, dtype=torch.float32, device=device)
    # Move models to the device
    model_regression.to(device)
    # Perform forward pass
    regression_result = model_regression(input_tensor_regression)

    # Compute gradients for each output w.r.t. pressure (second input)
    pressure_grads = []
    for i in range(regression_result.shape[1]):
        model_regression.zero_grad()
        regression_result[:, i].backward(retain_graph=True)
        # Collect the gradient for temperature only (index 0)
        pressure_grads.append(input_tensor_regression.grad[:, 2].clone())
        input_tensor_regression.grad.zero_()  # Reset gradients for the next output

    # Adjust gradients by multiplying with sigma
    adjusted_pressure_grads = [g * (sigma_y[i] / sigma_x[2]) for i, g in enumerate(pressure_grads)]

    # Return numpy array of adjusted gradients for pressure
    return 1, [grad.item() for grad in adjusted_pressure_grads]


def deriv_z_ABSORBENT(temperature, pressure, z_H2O, z_ABSORBENT, z_CO2, z_N2):
    global scalerXDataRegression
    global scalerYDataRegression
    global device
    global model_regression

    input_data = np.concatenate([
        np.array([temperature]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([pressure]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_H2O]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_ABSORBENT]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_CO2]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_N2]).reshape(1, 1)  # Reshape to (1, 1)
        # Reshape composition to have 1 row and -1 lets numpy decide the number of columns
    ], axis=1)
    # input_data_reshaped = input_data.reshape(1, -1)
    # Assuming scalerClassification and scalerRegression do not need to be moved to device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        input_data_scaled_regression = scalerXDataRegression.transform(input_data)

    # Convert scaled data to tensor, enable gradient, and move to device
    input_tensor_regression = torch.tensor(input_data_scaled_regression, dtype=torch.float32, device=device)
    input_tensor_regression.requires_grad = True

    # Get scaling factor for output and convert to tensor
    sigma_x = torch.tensor(scalerXDataRegression.scale_, dtype=torch.float32, device=device)
    sigma_y = torch.tensor(scalerYDataRegression.scale_, dtype=torch.float32, device=device)
    # Move models to the device
    model_regression.to(device)
    # Perform forward pass
    regression_result = model_regression(input_tensor_regression)

    # Compute gradients for each output w.r.t. pressure (second input)
    pressure_grads = []
    for i in range(regression_result.shape[1]):
        model_regression.zero_grad()
        regression_result[:, i].backward(retain_graph=True)
        # Collect the gradient for temperature only (index 0)
        pressure_grads.append(input_tensor_regression.grad[:, 3].clone())
        input_tensor_regression.grad.zero_()  # Reset gradients for the next output

    # Adjust gradients by multiplying with sigma
    adjusted_pressure_grads = [g * (sigma_y[i] / sigma_x[3]) for i, g in enumerate(pressure_grads)]

    # Return numpy array of adjusted gradients for pressure
    return 1, [grad.item() for grad in adjusted_pressure_grads]


def deriv_z_CO2(temperature, pressure, z_H2O, z_ABSORBENT, z_CO2, z_N2):
    global scalerXDataRegression
    global scalerYDataRegression
    global device
    global model_regression

    input_data = np.concatenate([
        np.array([temperature]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([pressure]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_H2O]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_ABSORBENT]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_CO2]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_N2]).reshape(1, 1)  # Reshape to (1, 1)
        # Reshape composition to have 1 row and -1 lets numpy decide the number of columns
    ], axis=1)
    # input_data_reshaped = input_data.reshape(1, -1)
    # Assuming scalerClassification and scalerRegression do not need to be moved to device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        input_data_scaled_regression = scalerXDataRegression.transform(input_data)

    # Convert scaled data to tensor, enable gradient, and move to device
    input_tensor_regression = torch.tensor(input_data_scaled_regression, dtype=torch.float32, device=device)
    input_tensor_regression.requires_grad = True

    # Get scaling factor for output and convert to tensor
    sigma_x = torch.tensor(scalerXDataRegression.scale_, dtype=torch.float32, device=device)
    sigma_y = torch.tensor(scalerYDataRegression.scale_, dtype=torch.float32, device=device)
    # Move models to the device
    model_regression.to(device)
    # Perform forward pass
    regression_result = model_regression(input_tensor_regression)

    # Compute gradients for each output w.r.t. pressure (second input)
    pressure_grads = []
    for i in range(regression_result.shape[1]):
        model_regression.zero_grad()
        regression_result[:, i].backward(retain_graph=True)
        # Collect the gradient for temperature only (index 0)
        pressure_grads.append(input_tensor_regression.grad[:, 4].clone())
        input_tensor_regression.grad.zero_()  # Reset gradients for the next output

    # Adjust gradients by multiplying with sigma
    adjusted_pressure_grads = [g * (sigma_y[i] / sigma_x[4]) for i, g in enumerate(pressure_grads)]

    # Return numpy array of adjusted gradients for pressure
    return 1, [grad.item() for grad in adjusted_pressure_grads]


def deriv_z_N2(temperature, pressure, z_H2O, z_ABSORBENT, z_CO2, z_N2):
    global scalerXDataRegression
    global scalerYDataRegression
    global device
    global model_regression

    input_data = np.concatenate([
        np.array([temperature]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([pressure]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_H2O]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_ABSORBENT]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_CO2]).reshape(1, 1),  # Reshape to (1, 1)
        np.array([z_N2]).reshape(1, 1)  # Reshape to (1, 1)
        # Reshape composition to have 1 row and -1 lets numpy decide the number of columns
    ], axis=1)
    # input_data_reshaped = input_data.reshape(1, -1)
    # Assuming scalerClassification and scalerRegression do not need to be moved to device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        input_data_scaled_regression = scalerXDataRegression.transform(input_data)

    # Convert scaled data to tensor, enable gradient, and move to device
    input_tensor_regression = torch.tensor(input_data_scaled_regression, dtype=torch.float32, device=device)
    input_tensor_regression.requires_grad = True

    # Get scaling factor for output and convert to tensor
    sigma_x = torch.tensor(scalerXDataRegression.scale_, dtype=torch.float32, device=device)
    sigma_y = torch.tensor(scalerYDataRegression.scale_, dtype=torch.float32, device=device)
    # Move models to the device
    model_regression.to(device)
    # Perform forward pass
    regression_result = model_regression(input_tensor_regression)

    # Compute gradients for each output w.r.t. pressure (second input)
    pressure_grads = []
    for i in range(regression_result.shape[1]):
        model_regression.zero_grad()
        regression_result[:, i].backward(retain_graph=True)
        # Collect the gradient for temperature only (index 0)
        pressure_grads.append(input_tensor_regression.grad[:, 5].clone())
        input_tensor_regression.grad.zero_()  # Reset gradients for the next output

    # Adjust gradients by multiplying with sigma
    adjusted_pressure_grads = [g * (sigma_y[i] / sigma_x[5]) for i, g in enumerate(pressure_grads)]

    # Return numpy array of adjusted gradients for pressure
    return 1, [grad.item() for grad in adjusted_pressure_grads]


# Example usage:
def classify_and_regress_validation(input_data):
    if len(input_data) == 0:
        return 0
    return 1


def regress_validation(input_data):
    if len(input_data) == 0:
        return 0
    return 1


def init(trial_number=0,
         timeStamp=TIMESTAMP):
    # set variables global for ???
    global scalerXDataRegression
    global scalerYDataRegression
    global scalerClassification
    global device
    global model_classification
    global model_regression

    # Define the base directory
    base_dir = './models'
    # Correctly join the paths without leading slashes in the file components
    loadPath_1 = os.path.join(base_dir, 'Training_' + timeStamp, 'scalerXData_regression.save').replace("\\", "/")
    loadPath_2 = os.path.join(base_dir, 'Training_' + timeStamp, 'scalerYData_regression.save').replace("\\", "/")
    scalerXDataRegression = joblib.load(loadPath_1)
    scalerYDataRegression = joblib.load(loadPath_2)

    scalerClassification = joblib.load('./models/scaler_classification.save')
    loadPath_3 = os.path.join(base_dir, 'Training_' + timeStamp, 'completed_trials_regression.csv').replace("\\", "/")

    hidden_size_regression = get_hidden_size(trial_number, loadPath_3)
    hidden_size_classification = pd.read_csv('./models/hidden_size_classification.csv')

    # Check for CUDA availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_classification = NeuralNetworkClassification(hidden_size_classification['hidden_size'][0]).to(device)

    state_dict_classification = torch.load('./models/ANNClassificationPytorch.pth')

    model_classification.load_state_dict(state_dict_classification)
    model_classification.eval()  # Set the model to inference mode

    model_regression = NeuralNetworkRegression(hidden_size_regression).to(device)

    loadPath_4 = os.path.join(base_dir, 'Training_' + timeStamp, 'bestTrials', 'direct',
                              f'best_model_trial_{trial_number}.pth').replace("\\", "/")
    loadPath_5 = os.path.join(base_dir, 'Training_' + timeStamp, 'bestTrials', 'afterContinuedTraining',
                              f'best_model_trial_{trial_number}.pth').replace("\\", "/")

    state_dict_regression = torch.load(loadPath_4)

    model_regression.load_state_dict(state_dict_regression)
    model_regression.eval()  # Set the model to inference mode


# initialise something
init(35, '2024_06_14_08_06_03')

# define variables to pass into SAFT via gPROMS
T = 313
p_ANN = np.linspace(1e5, 2.5e5, int(1e3))
pressure_ANN = 1.01e5
n_CO2_vector = np.linspace(0.5 * 0.0026446956, 2 * 0.0026446956, int(1e3))
z = np.array([0.028604876, 0.0032229, 0.0026446956, 0.01838627])
mw = [18.015, 61.08, 44.01, 28.01]

epsilon = 1e-2
output = regress(1 / T,
                 pressure_ANN,
                 z[0], z[1], z[2], z[3],
                 z[0] * mw[0], z[1] * mw[1], z[2] * mw[2], z[3] * mw[3])

result_x = pd.DataFrame(columns=["x_ABSORBENT", "x_CO2", "x_N2"])
result_y = pd.DataFrame(columns=["y_water", "y_ABSORBENT", "y_CO2"])

for n_CO2 in n_CO2_vector:
    z[2] = n_CO2
    z_new = z / np.sum(z)
    statusflag, output = regress(1 / T,
                                 pressure_ANN,
                                 z_new[0], z_new[1], z_new[2], z_new[3],
                                 z_new[0] * mw[0], z_new[1] * mw[1], z_new[2] * mw[2], z_new[3] * mw[3])

    # Splitting the output array
    output_x = output[:3]  # First 4 elements for result_x
    output_y = output[3:]  # Last 4 elements for result_y

    # Convert numpy arrays to DataFrames with matching column names
    df_x = pd.DataFrame([output_x], columns=result_x.columns)
    df_y = pd.DataFrame([output_y], columns=result_y.columns)

    result_x = pd.concat([result_x, df_x], ignore_index=True)
    result_y = pd.concat([result_y, df_y], ignore_index=True)

# Define variables to pass into SAFT via gPROMS
groupsGC = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
T = 313
p_ANN = np.linspace(1e5, 2.5e5, int(1e4))
pressure_SAFT = 1.01e5
n_CO2_vector = np.linspace(0.2 * 0.0026446956, 10 * 0.0026446956, int(1e3))
z = np.array([0.028604876, 0.0032229, 0.0026446956, 0.01838627])
print(z / np.sum(z))

gopython.start_only()
gopython.select("gOPythonSamanVariableGroupInput", "gOPythonSamanVariableGroupInput")
gopython.simulate("gOPythonSamanVariableGroupInput")

null = io.StringIO()

results_SAFT = []
x = 100
z_new = np.zeros([4, int(1e3)])

with tqdm(total=len(n_CO2_vector)) as pbar:
    for i, n_CO2 in enumerate(n_CO2_vector, start=0):
        # Redirect stdout and stderr to the null object
        with redirect_stdout(null), redirect_stderr(null):
            z[2] = n_CO2
            z_new[:, i] = z / np.sum(z)
            gPROMSInput = np.concatenate((np.array([T]), np.array([pressure_SAFT]), np.array(z_new[:, i]), groupsGC))
            status, result = gopython.evaluate(gPROMSInput)
            results_SAFT.append(np.concatenate((result[:2],
                                                result[2:6],
                                                result[6:10],
                                                result[10:14],
                                                result[14:16])
                                               )
                                )
            pass
        i = i + 1
        pbar.update(1)

gopython.stop()

print(np.array(results_SAFT).shape)
results_SAFT = np.array(results_SAFT)
co2_loading_SAFT = results_SAFT[:, 8] / results_SAFT[:, 7]
partial_pressure_SAFT = results_SAFT[:, 12] * pressure_SAFT

print(partial_pressure_SAFT)

plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10, 10))
plt.plot(result_x["x_CO2"] / result_x["x_ABSORBENT"], pressure_ANN * result_y["y_CO2"], label='ANN')
plt.plot(co2_loading_SAFT, partial_pressure_SAFT, label='SAFT-$\gamma$ Mie')  # Plot the data
plt.xlabel('CO$_2$ loading / mol mol$^{-1}$')  # Add x-axis label
plt.ylabel('$P_{\mathrm{CO_2}}$ \ Pa')  # Add y-axis label
plt.yscale('log')  # Set y-axis to log scale
plt.ylim([0.1, 1e7])
plt.xlim([0, 0.75])
plt.legend(loc='upper left', frameon=False, fontsize='large')
# Save the figure
plt.savefig('./graphics/CO2_loading.png', dpi=500)
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(z_new[2, :], result_x["x_CO2"], label="ANN")
plt.plot(z_new[2, :], results_SAFT[:, 8], label="SAFT")
plt.xlabel('z$_{\mathrm{CO_2}}$')  # Add x-axis label
plt.ylabel('x$_{\mathrm{CO_2}}$')  # Add y-axis label
plt.legend(loc='upper left', frameon=False, fontsize='large')
plt.savefig('./graphics/x_CO2.png', dpi=500)
plt.show()
