import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils.rnn import pad_sequence

from usefultools.useful_tools import *


def prep_for_dl(df: pd.DataFrame, lag_window: int, forward_window: int):
    """prepare dataset"""
    df["open_interest"] = df["open_interest"].pct_change()
    df["volume"] = df["volume"].pct_change()
    df = df.replace([np.inf, -np.inf], 0.0001)
    df.dropna(inplace=True)
    standardize(df)
    df.dropna(inplace=True)
    df = renaming(df)
    prep_label(df, "close", -forward_window, "one_hot")
    X, y = create_sequence_all(df, lag_window)
    X, y = torch.tensor(X, dtype=torch.float32), torch.Tensor(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=32
    )
    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, test_loader


def define_model(model: nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Instantiate the model
    model = model.to(device)

    return model


def train_eval_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim,
    train_loader,
    test_loader,
    accuracy_threshold: float,
    epochs: int = 500,
    verbose: bool = False,
    print_eval_every: int = 10,
):
    """
    Train and evaluate the model
    default epochs = 500
    required inputs are model, criterion, optimizer, train_loader, test_loader
    option to print the loss and accuracy, with print_every
    default accuracy_threshold is 0.65
    """

    torch.cuda.empty_cache()
    start_time = time.perf_counter()
    epoch_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in range(epochs):
        model.train()

        for i, (inputs, targets) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.argmax(dim=1) # convert one-hot to index
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % print_eval_every == 0:
            if verbose: print(f'Epoch: {epoch + 1}, loss: {loss.item():.4f}')

            # evaluate performance on test set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(test_loader, 0):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = round_to_ones(model(inputs))
                    identical_elements = (outputs == targets).all(dim=1)
                    correct += identical_elements.sum()

            total = len(test_loader.dataset)
            if verbose: print(f'Accuracy of the network on the test set: \
                  {(correct) / total * 100}%')
            if correct / total > accuracy_threshold:
                if verbose: print(f"model successfully trained, out of sample accuracy above {correct/total * 100}%")
                break
            if verbose: print(f'{epoch+1} Epochs took {time.perf_counter() - epoch_time:.4f} seconds')
            epoch_time = time.perf_counter()
    end_time = time.perf_counter()
    if verbose: print(f'Training took {end_time - start_time:.4f} seconds')
    return model


def prep_for_eval(df: pd.DataFrame):
    """prepare dataset"""
    df["open_interest"] = df["open_interest"].pct_change()
    df["volume"] = df["volume"].pct_change()
    df.drop(["turnover"], axis=1, inplace=True)
    standardize(df)
    df.dropna(inplace=True)
    x_cols = [col for col in df.columns]
    xs = np.stack([df[x_col].values for x_col in x_cols], axis=-1)
    X = torch.tensor(xs, dtype=torch.float32)
    X = X.view(1, X.shape[0], X.shape[1])
    return X


def save_model(model: nn.Module, name: str):
    if not os.path.exists("models"):
        os.makedirs("models")

    torch.save(model, f"models/{name}.pth")
    print("Model saved")


def collate_fn(batch):
    # Separate the sequence and label from the batch
    sequences, labels = zip(*batch)
    
    # Pad the sequences
    sequences_padded = pad_sequence([torch.as_tensor(s, device='cuda:0') for s in sequences], batch_first=True)
    
    return sequences_padded, torch.LongTensor(labels)