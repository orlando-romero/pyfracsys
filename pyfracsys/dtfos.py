from .fracdiff import fracdiff

import torch
import torch.nn as nn

import gc

class DTFOS(nn.Module):
    def __init__(self, data: list[torch.Tensor]) -> None: # TODO: allow data to be provided in different formats
        super().__init__()
                
        # Check that the input is valid
        if len(data) == 0:
            raise ValueError("`data' must be a non-empty list")
        for X in data:
            if not isinstance(X, torch.Tensor):
                raise TypeError("`data' must be a list of torch.Tensors")
            if X.dim() != 2:
                raise TypeError("`data' must be a list of 2D arrays")
            if X.shape[0] < X.shape[1]:
                raise TypeError("Every element in `data' must be T x n array with T >= n")
        self.device = data[0].device
        self.dtype = data[0].dtype
        for X in data:
            if X.device != self.device or X.dtype != self.dtype:
                raise ValueError("All tensors in `data' must have the same device and dtype")
        
        # Get the true (padded) dimensions of the data
        self.B = len(data)
        self.T = [data[b].shape[0] for b in range(self.B)]
        self.n = [data[b].shape[1] for b in range(self.B)]
        
        # Pad the data with zeros to make them all the same length
        self.T_max = max(self.T)
        self.n_max = max(self.n)
        self.X = torch.zeros(self.B, self.T_max, self.n_max).to(self.device, dtype=self.dtype)
        for b in range(self.B):
            self.X[b, :self.T[b], :self.n[b]] = data[b]
        
        # Store corr matrices R and their pseudo-inverses
        self.R = torch.zeros(self.B, self.n_max, self.n_max).to(self.device, dtype=self.dtype)
        self.R_inv = torch.zeros(self.B, self.n_max, self.n_max).to(self.device, dtype=self.dtype)
        for b in range(self.B):
            self.R[b,:,:] = self.X[b,:,:].T @ self.X[b,:,:]
            self.R_inv[b,:,:] = torch.linalg.pinv(self.R[b,:,:])
        
        # Store the model parameters, A and alpha
        self.A = nn.Parameter(torch.zeros(self.B, self.n_max, self.n_max).to(self.device, dtype=self.dtype))
        self.alpha = nn.Parameter(torch.ones(self.B, self.n_max).to(self.device, dtype=self.dtype))
        
        # Pre-compute and store the fracdiff of the stored data
        self.Y = fracdiff(self.X, self.alpha)
            
    def _MSE():
        return None
    

# def reshape_data(data: list[torch.Tensor]) -> torch.Tensor:
    