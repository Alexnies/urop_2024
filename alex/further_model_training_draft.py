import os
import torch
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model import RegressionModel


# create timestamp for saving models
now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

device = "cpu"

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

y_data_scalar = StandardScaler()
y_train_scaled = y_data_scalar.fit_transform(y_train)
y_val_scaled = y_data_scalar.transform(y_val)
y_test_scaled = y_data_scalar.transform(y_test)

# Convert data to PyTorch tensors and move to the chosen device
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)



# Results Plotting

MODEL_SAVE_PATH = '/home/an1821/Desktop/urop_2024/alex/experiments/2024_07_25_15_00_23/models/best_model_trial_68.pth'


def rename_state_dict_keys(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = 'layer_stack.' + key
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


# Load the model
model = RegressionModel(activation_l0='elu',
                        activation_l1='elu',
                        activation_l2='elu',
                        hidden_size_l0=500,
                        hidden_size_l1=500,
                        hidden_size_l2=500)
old_state_dict = torch.load(MODEL_SAVE_PATH)
new_state_dict = rename_state_dict_keys(old_state_dict)
model.load_state_dict(new_state_dict)


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


plot_prediction_comparison(model,
                           labels=['x_absorbent', 'x_co2', 'x_n2', 'y_water', 'y_absorbent', 'y_co2'])
