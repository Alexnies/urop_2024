import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


def save_dataset(X: pd.DataFrame,
                 y: pd.DataFrame,
                 test_size: float = 0.2,
                 random_seed: int = 42,
                 name: str = None):
    # create data directory
    dataset_path = os.path.join(os.getcwd(), "training_data")
    os.makedirs(dataset_path, exist_ok=True)

    # create training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    # save datasets
    if name is not None:
        X_train.to_csv(os.path.join(dataset_path, f"X_train_{name}.csv"), index=False)
        X_test.to_csv(os.path.join(dataset_path, f"X_test_{name}.csv"), index=False)
        y_train.to_csv(os.path.join(dataset_path, f"y_train_{name}.csv"), index=False)
        y_test.to_csv(os.path.join(dataset_path, f"y_test_{name}.csv"), index=False)
    else:
        X_train.to_csv(os.path.join(dataset_path, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(dataset_path, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(dataset_path, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(dataset_path, "y_test.csv"), index=False)


def save_best_model(model,
                    timestamp: str,
                    model_name: str):
    # create models directory
    model_path = os.path.join(os.getcwd(), "models", timestamp)
    os.makedirs(model_path, exist_ok=True)

    # create model save path
    model_name = model_name + ".pth"
    model_save_path = os.path.join(model_path, model_name)
    torch.save(obj=model.state_dict(),
               f=model_save_path)
