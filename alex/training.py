import torch
from tqdm import tqdm


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
        if batch % 200 == 0:
            print(f"Looked at {batch * len(X)}/{len(data_loader.dataset)} samples.")

    # divide total train loss by length of train dataloader
    train_loss /= len(data_loader)
    print(f"Train loss: {train_loss:.5f}")


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
        print(f"Validation loss: {val_loss:.5f}")
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
    print(f"Test loss: {test_loss:.5f}")
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
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # accumulate the loss and acc values per batch
            loss += criterion(y_pred, y)

        # scale loss to find the average loss per batch
        loss /= len(data_loader)

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": loss.item()}
