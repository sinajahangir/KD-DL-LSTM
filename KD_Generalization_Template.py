# -*- coding: utf-8 -*-
"""
The following code can be used for knowledge distillation (KD) of deep learning (DL) models.
The goal is to use KD to transfer knowledge from a DL model trained with a different, and potentiallt better data,
to a DL model that will use alternative input.

Daymet and ERA5-Land were used as input to the teacher and student models

The resulted model shows subtantian improvement compared to a "vanilla" DL model (training using the different source), where KD
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
    def __init__(self, data1,data2, targets, seq_length):
        """
        Args:
            data (np.ndarray or torch.Tensor): The 2D array of shape [num_samples, num_features].
            targets (np.ndarray or torch.Tensor): The 1D array of target values.
            seq_length (int): The length of the sequence for LSTM.
        """
        self.data1 = torch.tensor(data1, dtype=torch.float32)
        self.data2 = torch.tensor(data2, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data1) - self.seq_length + 1

    def __getitem__(self, idx):
        sequence1 = self.data1[idx: idx + self.seq_length,:]
        sequence2 = self.data2[idx: idx + self.seq_length,:]
        target = self.targets[idx + self.seq_length - 1]  # Align target with the end of the sequence
        return sequence1,sequence2, target
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
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.4):
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
        return out, lstm_out[:, -1, :]

# Define the Knowledge Distillation framework
class KnowledgeDistillation(nn.Module):
    def __init__(self, student, teacher, beta=0.1):
        super(KnowledgeDistillation, self).__init__()
        self.student = student
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False  # Freeze teacher weights
        self.beta = beta  # Weight for embedding alignment loss

    def forward(self, x1,x2):
        x_student = x2
        student_out, student_hidden = self.student(x_student)
        with torch.no_grad():
            _, teacher_hidden = self.teacher(x1)
        return student_out, student_hidden, teacher_hidden
    
    

# Loss function with embedding alignment
class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, student_out, student_hidden, teacher_hidden, true_values, beta):
        loss = (self.mse(student_out, true_values) +
                beta * self.mse(student_hidden, teacher_hidden))
        return loss
        
#%%
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
# Model initialization
input_size = 32  # number of input features for teacher (Daymet 5+static 27)
input_size_student=64 # number of input features for student (ERA 37+static 27)
hidden_size = 256
num_layers = 1
output_size = 1
dropout_prob = 0.4

#define theacher model
teacher_model = LSTMModel(input_size, hidden_size, num_layers, output_size,dropout_prob)
#load teacher model
#change this
teacher_model.load_state_dict(torch.load('ModelLSTM_st_random_Final_seed_%d_All'%(seed),weights_only=True))
student_model = LSTMModel(input_size_student, hidden_size, num_layers, output_size,dropout_prob)



# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)
#hyperparamter controlling the tradeoff between loss function components
beta=50
distill_model = KnowledgeDistillation(student_model, teacher_model, beta=beta)
criterion = DistillationLoss()
optimizer = optim.Adam(distill_model.student.parameters(), lr=0.001)
distill_model.to(device)
#%%
max_epochs_trained = 40
patience = 4
best_val_loss = float('inf')
early_stop_counter = 0
epochs_trained = 0
scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.1, min_lr=1e-6)
loss_fn =nn.MSELoss()
best_model_state_list=[]
for epoch in range(max_epochs_trained):
    print('new epoch')
    print(epoch)
    distill_model.train()
    epoch_loss = 0.0
    for X_batch,X_batch_era, Y_batch in dataloader_train: #The training data loader.
    #This should be defined using the generator
        # Move data to the same device as the model
        X_batch, X_batch_era, Y_batch = X_batch.to(device),X_batch_era.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        student_out, student_hidden, teacher_hidden = distill_model(X_batch,X_batch_era)
        
        loss = criterion(student_out, student_hidden, teacher_hidden, Y_batch, distill_model.beta)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Compute average training loss
    epoch_loss /= len(dataloader_train.dataset)
    #print("Sum of teacher weights:", sum(p.sum() for p in distill_model.teacher.parameters()))
    # Validation every 10 epochs
    if epoch % 2 == 0:
        distill_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for XX_batch, XX_batch_era, YY_batch in dataloader_val: #The validation data loader.
            #This should be defined using the generator
                # Move data to the same device as the model
                XX_batch,XX_batch_era, YY_batch = XX_batch.to(device),XX_batch_era.to(device), YY_batch.to(device)

                # Forward pass for validation
                student_out, student_hidden, teacher_hidden = distill_model(XX_batch,XX_batch_era)
                loss_ = loss_fn(student_out,YY_batch)
                val_loss += loss_.item()

        # Compute average validation loss
        val_loss /= len(dataloader_val.dataset)

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state_list.append(distill_model.state_dict())
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

if best_model_state_list:
    student_model.load_state_dict(best_model_state_list[-1])
    print("Best model restored!")
torch.save(best_model_state_list[-1], 'NAMEOFINTEREST')# chnage this