# -*- coding: utf-8 -*-
"""
The following code can be used for knowledge distillation (KD) of deep learning (DL) models.
The goal is to use KD to transfer knowledge from an ensmble of DL models, here a pool of LSTM models,
to a single DL model.
The resulted model shows subtantian improvement compared to a "vanilla" DL model, where KD
was not utilized for training.
First version: May 2025 (Sina Jahangir)
"""
#%%
#import necessary libraries
import torch
import torch.nn as nn
import numpy as np
#%%
# Data generator for creating 3D inputs to DL models
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Custom dataset for handling large 2D arrays and converting them to LSTM-ready 3D sequences.
    """
    def __init__(self, data, targets, seq_length):
        """
        Args:
            data (np.ndarray or torch.Tensor): The 2D array of shape [num_samples, num_features].
            targets (np.ndarray or torch.Tensor): The 1D array of target values.
            seq_length (int): The length of the sequence for LSTM.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        sequence = self.data[idx: idx + self.seq_length,:]
        target = self.targets[idx + self.seq_length - 1]  # Align target with the end of the sequence
        return sequence, target
#%%
# set options
seq_length = 365
batch_size = 1024
import random
seed=213 # change this
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For CUDA
np.random.seed(seed)  # For NumPy
random.seed(seed)  # For Python's random module
torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
#%%
# LSTM base model (student and teacher both have the same structure here)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        out = self.dropout(lstm_out[:, -1, :]) # take the last time step output
        # out shape: (batch_size, hidden_size)
        out = self.linear(out)
        # out shape: (batch_size, output_size)
        return out

class TeacherEnsemble(nn.Module):
    def __init__(self, base_models):
        super(TeacherEnsemble, self).__init__()
        self.lstm_models = nn.ModuleList(base_models)
        for model in self.lstm_models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            model_outputs = [model(x) for model in self.lstm_models]
            stacked_outputs = torch.stack(model_outputs, dim=0)
            averaged_output = torch.mean(stacked_outputs, dim=0)
        return averaged_output
import torch.nn.functional as F

# Distillation loss
def regression_distillation_loss(student_outputs, true_labels, teacher_outputs, alpha):
    """
    Calculates the combined loss for regression-based knowledge distillation.
    Optimizing for MSE is equivalent to optimizing for RMSE.
    
    Args:
        student_outputs: Predictions from the student model.
        true_labels: The ground truth continuous values (hard targets).
        teacher_outputs: Averaged predictions from the teacher ensemble (soft targets).
        alpha (float): Weight to balance the two loss components.
    """
    # Ensure labels have the same shape as outputs, e.g., (batch_size, 1)
    if true_labels.ndim == 1:
        true_labels = true_labels.unsqueeze(1)
        
    # 1. Distillation Loss (MSE between student and teacher)
    # This encourages the student to mimic the teacher's output.
    distillation_loss = F.mse_loss(student_outputs, teacher_outputs)

    # 2. Student Loss (MSE between student and true labels)
    # This is the standard regression loss against the ground truth.
    student_loss = F.mse_loss(student_outputs, true_labels)
    
    # 3. Combine the two losses
    # The student learns from both the teacher's knowledge and the true data.
    combined_loss = alpha * distillation_loss + (1.0 - alpha) * student_loss
    
    return combined_loss
        
#%%
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

# Model parameters
input_size = 32  # Dynamic (5)+ static(27)
hidden_size = 256
num_layers = 1
output_size = 1
dropout_prob = 0.4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_base_models(seed_list):
    """Loads pre-trained models from disk."""
    loaded_models = []
    for i in seed_list:
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)
        model.to(device)
        # 5 regional LSTM models. Identical structure optimized with different initialization
        # change this
        model.load_state_dict(torch.load('ModelLSTM_st_random_Final_seed_%d_All'%(i),weights_only=True))
        loaded_models.append(model)
        del model
    print("Base models loaded.\n")
    return loaded_models

base_models = load_base_models([113,213,313,413,513])
teacher_model=TeacherEnsemble(base_models)

student_model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)
student_model.to(device)
#%%
#Training loop
max_epochs_trained = 40
patience = 4
best_val_loss = float('inf')
early_stop_counter = 0
epochs_trained = 0
optimizer = optim.Adam(student_model.parameters(),lr=1e-3)
scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.1, min_lr=1e-6)
loss_fn =nn.MSELoss()
best_model_state_list=[]
#hyperparamter controlling the tradeoff between loss function components
ALPHA=0.5
for epoch in range(max_epochs_trained):
    print('new epoch')
    print(epoch)
    student_model.train()
    epoch_loss = 0.0
    for X_batch, Y_batch in dataloader_train: #The training data loader.
    #This should be defined using the generator
        # Move data to the same device as the model
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Get teacher's soft targets
        teacher_outputs = teacher_model(X_batch)
    
    # Get student's predictions
        student_outputs = student_model(X_batch)
    
    # Calculate the combined regression distillation loss
        loss = regression_distillation_loss(
            student_outputs=student_outputs,
            true_labels=Y_batch,
            teacher_outputs=teacher_outputs,
            alpha=ALPHA
        )
    
        # Standard optimization steps for the student
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    # Compute average training loss
    epoch_loss /= len(dataloader_train.dataset)

    # Validation every 2 epochs
    if epoch % 2 == 0:
        student_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for XX_batch, YY_batch in dataloader_val: #The validation data loader.
            #This should be defined using the generator
                # Move data to the same device as the model
                XX_batch, YY_batch = XX_batch.to(device), YY_batch.to(device)

                # Forward pass for validation
                y_pred = student_model(XX_batch)
                # for validation RMSE was used. This can be changed
                loss_ = loss_fn(y_pred, YY_batch)
                val_loss += loss_.item()

        # Compute average validation loss
        val_loss /= len(dataloader_val.dataset)

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state_list.append(student_model.state_dict())
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break

        print(f"Epoch {epoch}: Train Loss: {epoch_loss:.5f}, Val Loss: {val_loss:.5f}")
        scheduler.step(val_loss)
        print(scheduler.get_last_lr())
    epochs_trained += 1
    # Step the scheduler


    if epochs_trained >= max_epochs_trained:
        print(f"Reached maximum of {max_epochs_trained} training epochs.")
        break

# Restore the best model
if best_model_state_list:
    student_model.load_state_dict(best_model_state_list[-1])
    print("Best model restored!")
torch.save(best_model_state_list[-1], 'NAMEOFINTEREST')# chnage this