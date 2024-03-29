from .fracdiff import fracdiff
from .bisection import bisection_grid

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
        
        # Store correlation matrices and their pseudo-inverses
        self.R = torch.zeros(self.B, self.n_max, self.n_max).to(self.device, dtype=self.dtype)
        self.R_inv = torch.zeros(self.B, self.n_max, self.n_max).to(self.device, dtype=self.dtype)
        for b in range(self.B):
            self.R[b,:,:] = self.X[b,:,:].T @ self.X[b,:,:]
            self.R_inv[b,:,:] = torch.linalg.pinv(self.R[b,:,:])
        
        # Store the model parameters, A and alpha (nn.Parameter uses extra memory!)
        self.A = torch.zeros(self.B, self.n_max, self.n_max).to(self.device, dtype=self.dtype)
        self.alpha = torch.ones(self.B, self.n_max).to(self.device, dtype=self.dtype)
        
        # Pre-compute and store the fracdiff of the stored data
        self.Y = fracdiff(self.X, self.alpha)
            
    def _MSE(self) -> torch.Tensor:
        E = self.resid()
        return torch.sum(E**2, dim=1)
    
    def resid(self) -> torch.Tensor:
        return self.Y[:,1:,:] - torch.bmm(self.X[:,:-1,:], self.A.transpose(1,2))
    
    def fit_A(self) -> None:
        C = torch.bmm(self.Y[:,1:,:].transpose(1,2), self.X[:,:-1,:])    
        self.A = torch.bmm(C, self.R_inv)
        
    def fit(self,
            alpha_min:float = 0.0,
            alpha_max:float = 1.0,
            num_grids:int = 2,
            max_iter:int = 12,
            tol:float = 1e-4) -> None:
        
        # Check that the input is valid
        if not isinstance(alpha_min, float):
            raise TypeError("`alpha_min' must be a float")
        if not isinstance(alpha_max, float):
            raise TypeError("`alpha_max' must be a float")
        if not isinstance(num_grids, int):
            raise TypeError("`num_grids' must be an int")
        if not isinstance(max_iter, int):
            raise TypeError("`max_iter' must be an int")
        if not isinstance(tol, float):
            raise TypeError("`tol' must be a float")
        
        if alpha_min < 0.0:
            raise ValueError("`alpha_min' must be non-negative")
        if alpha_max < 0.0:
            raise ValueError("`alpha_max' must be non-negative")
        if alpha_min > alpha_max:
            raise ValueError("`alpha_min' must be less than or equal to `alpha_max'")
        if num_grids < 1:
            raise ValueError("`num_grids' must be at least 1")
        if max_iter < 1:
            raise ValueError("`max_iter' must be at least 1")
        if tol <= 0.0:
            raise ValueError("`tol' must be positive")
        
        def cost_fun(alpha: torch.Tensor) -> torch.Tensor:
            self.alpha = alpha
            self.Y = fracdiff(self.X, alpha) # TODO: force Y to be recomputed when alpha is reassigned
            self.fit_A()
            return self._MSE()
        
        alpha_min = torch.tensor(alpha_min).to(self.device, dtype=self.dtype)
        alpha_max = torch.tensor(alpha_max).to(self.device, dtype=self.dtype)
        
        self.alpha = bisection_grid(
            cost_fun,
            alpha_min.repeat(self.B, self.n_max),
            alpha_max.repeat(self.B, self.n_max),
            num_grids,
            max_iter,
            tol
            )
        
        self.fit_A()