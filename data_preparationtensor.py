import torch
import numpy
from torch.utils.data import DataLoader, TensorDataset

x_train_tensor = torch.as_tensor(X_train).float()
y_train_tensor = torch.as_tensor(y_train).float()

#Build Dataset
train_data = TensorDataset(x_train_tensor, y_train_tensor)

#Build DataLoader

train_loader = DataLoader(dataset = train_data, batch_size = 16,
shuffle = True)
