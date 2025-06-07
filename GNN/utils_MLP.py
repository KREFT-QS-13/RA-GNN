"""
Filename: utils_MLP.py
Description: Module containing functions for MLP-based prediction of nearest-neighbor correlations.
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from typing import List, Tuple

class EdgeMLP(nn.Module):
    """
    MLP architecture for predicting nearest-neighbor correlations.
    Similar to the Edge Prediction Head from the PNA GNN but without graph structure.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super(EdgeMLP, self).__init__()
        
        # Input dimension is num_deltas (number of time steps/history)
        self.input_dim = input_dim
        
        # Create MLP layers
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        self.double()  # Use double precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            x: Input tensor of shape [batch_size * num_edges, input_dim]
               containing nearest-neighbor correlations
            
        Returns:
            Predicted correlation differences of shape [batch_size * num_edges, 1]
        """
        return self.mlp(x)

def prepare_mlp_data(num_realizations: List[int], Ls: List[str], num_deltas: int, *, incl_scnd: bool = False, 
                     trgt_diff: bool = True, meas_basis: str = "Z",
                     data_folder: str = "./dataset_mps_NNN", datasets: List[str] = ["training", "validation", "test"], 
                     batch_sizes: List[int] = [32, 32, 32], dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare data for MLP training by extracting nearest-neighbor correlations.
    
    Args:
        data_list: List of Data objects from the dataset
        num_deltas: Number of time steps/history
        
    Returns:
        Tuple of (input_features, target_labels)
        - input_features: Tensor of shape [total_edges, num_deltas]
        - target_labels: Tensor of shape [total_edges, 1]
    """
    train_loader, validation_loader, test_loader = None, None, None

    for it in range(len(datasets)):
        data_list = []

        for L in Ls:
            Lx, Ly = L.split('x')
            Lx, Ly = int(Lx), int(Ly)
            

def create_mlp_dataloader(data_list: List[Data], batch_size: int, num_deltas: int) -> DataLoader:
    """
    Create a DataLoader for MLP training.
    
    Args:
        data_list: List of Data objects from the dataset
        batch_size: Batch size for training
        num_deltas: Number of time steps/history
        
    Returns:
        DataLoader for MLP training
    """
    input_features, target_labels = prepare_mlp_data(data_list, num_deltas)
    
    # Create a simple dataset class for the MLP
    class MLPDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
            
        def __len__(self):
            return len(self.features)
            
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    # Create dataset and dataloader
    dataset = MLPDataset(input_features, target_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_mlp(model: EdgeMLP, 
             train_loader: DataLoader, 
             val_loader: DataLoader, 
             num_epochs: int, 
             learning_rate: float = 1e-3,
             weight_decay: float = 1e-5) -> Tuple[List[float], List[float]]:
    """
    Train the MLP model.
    
    Args:
        model: EdgeMLP model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        
    Returns:
        Tuple of (training_losses, validation_losses)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        train_losses.append(epoch_train_loss / len(train_loader))
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
        
        val_losses.append(epoch_val_loss / len(val_loader))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_losses[-1]:.6f}, "
                  f"Val Loss: {val_losses[-1]:.6f}")
    
    return train_losses, val_losses 