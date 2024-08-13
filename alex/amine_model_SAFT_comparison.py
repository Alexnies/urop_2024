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

# import custom functions
from custom import CUDANotAvailableError, setup_cuda_as_device
from model import RegressionModel

# Define global variables
TIMESTAMP = "2024_08_12_22_48_59"
FURTHER_MODEL_DIR_PATH = os.path.join(os.getcwd(), 'experiments', TIMESTAMP, 'further_trained_model')
MODEL_SAVE_PATH = os.path.join(FURTHER_MODEL_DIR_PATH, 'model.pth')


def main():

    init()

    # define variables to pass into SAFT via gPROMS
    temp = 313
    pressure = 1.01e5
    n_CO2_vector = np.linspace(0.5 * 0.0026446956, 2 * 0.0026446956, int(1e3))
    n = np.array([0.028604876, 0.0032229, 0.0026446956, 0.01838627])

    result_x = pd.DataFrame(columns=["x_absorbent", "x_co2", "x_n2"])
    result_y = pd.DataFrame(columns=["y_water", "y_absorbent", "y_co2"])

    z_ANN = np.zeros([4, int(1e3)])

    for i, n_CO2 in enumerate(n_CO2_vector, start=0):
        n[2] = n_CO2
        z_ANN[:, i] = n / np.sum(n)
        output = regress(temp, pressure, z_ANN[0, i], z_ANN[1, i], z_ANN[2, i], z_ANN[3, i])

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
    n_CO2_vector = np.linspace(0.2 * 0.0026446956, 10 * 0.0026446956, int(1e3))

    gopython.start_only()
    gopython.select("gOPythonSamanVariableGroupInput",
                    "gOPythonSamanVariableGroupInput")  # first is process model, second is pw (default)
    gopython.simulate("gOPythonSamanVariableGroupInput")

    null = io.StringIO()  # redirect the output - for speed

    results_SAFT = []
    z_SAFT = np.zeros([4, int(1e3)])

    with tqdm(total=len(n_CO2_vector)) as pbar:
        for i, n_CO2 in enumerate(n_CO2_vector, start=0):
            # Redirect stdout and stderr to the null object
            with redirect_stdout(null), redirect_stderr(null):
                n[2] = n_CO2
                z_SAFT[:, i] = n / np.sum(n)  # now convert n to z
                gPROMSInput = np.concatenate(
                    (np.array([temp]), np.array([pressure]), np.array(z_SAFT[:, i]), groupsGC))
                status, result = gopython.evaluate(gPROMSInput)
                results_SAFT.append(np.concatenate((result[:2],  # T,P
                                                    result[2:6],  # global comp.
                                                    result[6:10],  # vapour/liq comp.
                                                    result[10:14],  # liq/vap
                                                    result[14:16])  # pure liq/vap identifier
                                                   )
                                    )
                pass
            i = i + 1
            pbar.update(1)

    gopython.stop()

    results_SAFT = np.array(results_SAFT)

    plot_prediction_comparison(result_x,
                               result_y,
                               results_SAFT,
                               pressure,
                               z_ANN,
                               z_SAFT)

def init():
    global X_data_scalar
    global y_data_scalar
    global model

    # Load the StandardScalar classes
    xscalar_save_path = os.path.join(FURTHER_MODEL_DIR_PATH, 'X_data_scalar.save')
    yscalar_save_path = os.path.join(FURTHER_MODEL_DIR_PATH, 'y_data_scalar.save')
    X_data_scalar = joblib.load(xscalar_save_path)
    y_data_scalar = joblib.load(yscalar_save_path)

    #hidden_size_regression = get_hidden_size(trial_number, loadPath_3)
    #hidden_size_classification = pd.read_csv('./models/hidden_size_classification.csv')

    # Load the model
    model = define_model()
    model.eval()  # Set the model to inference mode


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
    state_dict = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(state_dict)

    return model


def regress(temperature, pressure, z_water, z_absorbent, z_co2, z_n2):

    X_data = np.concatenate([
        np.array([1/temperature]).reshape(1, 1),
        np.array([pressure]).reshape(1, 1),
        np.array([z_water]).reshape(1, 1),
        np.array([z_absorbent]).reshape(1, 1),
        np.array([z_co2]).reshape(1, 1),
        np.array([z_n2]).reshape(1, 1)
        ], axis=1)

    # scale the X_data to pass into the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        X_data_scaled = X_data_scalar.transform(X_data)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_data_scaled, dtype=torch.float32)

    # put the model in eval mode
    model.eval()
    with torch.inference_mode():
        y_pred = model(X_tensor)

    # inverse scale the dataset
    y_pred = y_data_scalar.inverse_transform(y_pred)

    # Convert the NumPy arrays to pandas DataFrames
    y_pred_df = pd.DataFrame(y_pred,
                             columns=["x_absorbent", "x_co2", "x_n2", "y_water", "y_absorbent", "y_co2"])

    # Use the .clip() method to set all values below 0 to 0 in both DataFrames
    y_pred_df = y_pred_df.clip(lower=0)

    x_values = ["x_absorbent", "x_co2", "x_n2"]
    y_values = ["y_water", "y_absorbent", "y_co2"]

    output = pd.concat([y_pred_df[x_values], y_pred_df[y_values]], axis=1)

    return output.values.tolist()[0]


def plot_prediction_comparison(result_x,
                               result_y,
                               results_SAFT,
                               pressure,
                               z_ANN,
                               z_SAFT,
                               show: bool = False):

    co2_loading_SAFT = results_SAFT[:, 8] / results_SAFT[:, 7]
    partial_pressure_SAFT = results_SAFT[:, 12] * pressure

    plt.rcParams.update({'font.size': 20})

    plt.figure(figsize=(10, 10))
    plt.plot(result_x["x_co2"] / result_x["x_absorbent"], pressure * result_y["y_co2"], label='ANN')
    plt.plot(co2_loading_SAFT, partial_pressure_SAFT, label='SAFT-$\gamma$ Mie')  # Plot the data
    plt.xlabel('CO$_2$ loading / mol mol$^{-1}$')  # Add x-axis label
    plt.ylabel('$P_{\mathrm{CO_2}}$ \ Pa')  # Add y-axis label
    plt.yscale('log')  # Set y-axis to log scale
    plt.ylim([0.1, 1e7])
    plt.xlim([0, 0.75])
    plt.legend(loc='upper left', fontsize='large')

    figures_dir = os.path.join(FURTHER_MODEL_DIR_PATH, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    fig_name = 'co2_loading.png'
    save_path = os.path.join(figures_dir, fig_name)
    plt.savefig(save_path, dpi=500)

    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.plot(z_ANN[2, :], result_x["x_co2"], label="ANN")
    plt.plot(z_SAFT[2, :], results_SAFT[:, 8], label="SAFT")
    plt.xlabel('z$_{\mathrm{CO_2}}$')  # Add x-axis label
    plt.ylabel('x$_{\mathrm{CO_2}}$')  # Add y-axis label
    plt.legend(loc='lower right', fontsize='large')

    fig_name = 'z_vs_x_co2.png'
    save_path = os.path.join(figures_dir, fig_name)
    plt.savefig(save_path, dpi=500)

    if show:
        plt.show()
    plt.close()

main()
