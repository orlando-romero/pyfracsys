from .fracdiff import fracdiff
from .bisection import bisection_grid

import torch
import torch.nn as nn

from torch.nn.functional import relu

class DTFOS(nn.Module):
    def __init__(self, data: list[torch.Tensor], normalize_data:bool=True) -> None: # TODO: allow data to be provided in different formats
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
        
        self.B = len(data)
        self.T = [data[b].shape[0] for b in range(self.B)]
        self.n = [data[b].shape[1] for b in range(self.B)]
        
        # Store the data, padding it with zeros to make the num of observations and channels consistent
        self.T_max = max(self.T)
        self.n_max = max(self.n)
        self.X = torch.zeros(self.B, self.T_max, self.n_max).to(self.device, dtype=self.dtype)
        for b in range(self.B):
            self.X[b, :self.T[b], :self.n[b]] = data[b]
            if normalize_data:
                X_mean = torch.mean(self.X[b, :self.T[b], :self.n[b]], dim=0, keepdim=True)
                X_std = torch.std(self.X[b, :self.T[b], :self.n[b]], dim=0, keepdim=True)
                self.X[b, :self.T[b], :self.n[b]] = (self.X[b, :self.T[b], :self.n[b]] - X_mean) / X_std
        
        if normalize_data:
            for b in range(self.B):
                X_mean = torch.mean(self.X[b, :self.T[b], :self.n[b]], dim=0, keepdim=True)
                X_std = torch.std(self.X[b, :self.T[b], :self.n[b]], dim=0, keepdim=True)
                self.X[b, :self.T[b], :self.n[b]] = (self.X[b, :self.T[b], :self.n[b]] - X_mean) / X_std
        
        self.R_inv = None
        
        self.A = nn.Parameter(torch.zeros(self.B, self.n_max, self.n_max).to(self.device, dtype=self.dtype))
        self.alpha = nn.Parameter(torch.ones(self.B, self.n_max).to(self.device, dtype=self.dtype))
            
    def _MSE(self) -> torch.Tensor:
        E = self._resid()
        return torch.sum(E**2, dim=1)
    
    def _resid(self) -> torch.Tensor:
        Y = fracdiff(self.X, relu(self.alpha))
        return Y[:,1:,:] - torch.bmm(self.X[:,:-1,:], self.A.transpose(1,2))
    
    def MSE_loss(self): # TODO: refactor this
        loss_fun = nn.MSELoss()
        Y = fracdiff(self.X, relu(self.alpha))[:,1:,:]
        Yhat = torch.bmm(self.X[:,:-1,:], self.A.transpose(1,2))
        return loss_fun(Y, Yhat)
    
    def fit_A(self) -> None:
        if self.R_inv is None:
            R = torch.zeros(self.B, self.n_max, self.n_max).to(self.device, dtype=self.dtype)
            self.R_inv = torch.zeros(self.B, self.n_max, self.n_max).to(self.device, dtype=self.dtype)
            for b in range(self.B):
                R[b,:,:] = self.X[b,:-1,:].T @ self.X[b,:-1,:]
                self.R_inv[b,:,:] = torch.linalg.pinv(R[b,:,:])
                
        Y = fracdiff(self.X, relu(self.alpha))
        C = torch.bmm(Y[:,1:,:].transpose(1,2), self.X[:,:-1,:])
        self.A.data = torch.bmm(C, self.R_inv)
        
    def MSE(self) -> torch.Tensor:
        mse = self._MSE()
        return [mse[b, :self.n[b]] for b in range(self.B)]
    
    def resid(self) -> torch.Tensor:
        resid = self._resid()
        return [resid[b, :self.T[b]-1, :self.n[b]] for b in range(self.B)]
    
    def fit(self, method="OLS", **kwargs):
        if method == "OLS":
            self.fit_OLS(**kwargs)
        elif method == "LASSO":
            self.fit_LASSO(**kwargs)
        else:
            raise ValueError("`method' must be 'OLS' or 'LASSO'")
        
    def fit_LASSO(self):
        print("not yet implemented!")
        pass
        
    def fit_OLS(self,
            alpha_min:float = 0.0,
            alpha_max:float = 1.0,
            num_grids:int = 2,
            max_iter:int = 12,
            tol:float = 1e-4) -> None:
        
        # TODO: refactor
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
        
        with torch.no_grad():
            def cost_fun(alpha: torch.Tensor) -> torch.Tensor:
                self.alpha.data = alpha
                self.fit_A()
                return self._MSE()
            
            alpha_min = torch.tensor(alpha_min).to(self.device, dtype=self.dtype)
            alpha_max = torch.tensor(alpha_max).to(self.device, dtype=self.dtype)
            
            self.alpha.data = bisection_grid(
                cost_fun,
                alpha_min.repeat(self.B, self.n_max),
                alpha_max.repeat(self.B, self.n_max),
                num_grids,
                max_iter,
                tol
                )
            
            self.fit_A()