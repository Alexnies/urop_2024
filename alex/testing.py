import torch
from torch import nn
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from model import RegressionModel
from plotting import plot_prediction_comparison

# create timestamp for saving models
now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

device="cpu"

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

# Convert data to PyTorch tensors and move to the chosen device
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

model = RegressionModel(activation1='elu',
                        activation2='elu',
                        activation3='elu',
                        hidden_size=500)

PATH='/home/an1821/Desktop/urop_2024/alex/models/2024_07_20_11_09_13/best_trials/direct/best_model_trial_407.pth'
model.load_state_dict(torch.load(PATH))

plot_prediction_comparison(model_load_path=PATH,
                           X_test=X_test_tensor,
                           y_test=y_test_tensor,
                           timestamp=timestamp,
                           labels=['x_absorbent', 'x_co2', 'x_n2', 'y_water', 'y_absorbent', 'y_co2'])
