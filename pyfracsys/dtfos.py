from .fracdiff import fracdiff, fracdiff_kernel
from .bisection import bisection_grid

import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.optim import SGD, Adam

from torchaudio.functional import fftconvolve

from tqdm import tqdm
import matplotlib.pyplot as plt

class DTFOS(nn.Module):
    def __init__(self, data: list[torch.Tensor], normalize_data:bool=True, train_ratio:float=0.8) -> None: # TODO: allow data to be provided in different formats
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
        
        # self.T_train = [int(round(self.T[b] * train_ratio)) for b in range(self.B)]
        
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
                
        self.T_train = int(round(self.T_max * train_ratio)) # TODO: refactor
        
        if normalize_data:
            for b in range(self.B):
                X_mean = torch.mean(self.X[b, :self.T[b], :self.n[b]], dim=0, keepdim=True)
                X_std = torch.std(self.X[b, :self.T[b], :self.n[b]], dim=0, keepdim=True)
                self.X[b, :self.T[b], :self.n[b]] = (self.X[b, :self.T[b], :self.n[b]] - X_mean) / X_std
        
        self.R_inv = None
        
        self.A = nn.Parameter(torch.zeros(self.B, self.n_max, self.n_max).to(self.device, dtype=self.dtype))
        self.alpha = nn.Parameter(torch.ones(self.B, self.n_max).to(self.device, dtype=self.dtype))
            
    def _MSE(self, X=None) -> torch.Tensor:
        if X is None:
            X = self.X
        E = self._resid(X)
        return torch.sum(E**2, dim=1) / E.shape[1]
    
    def _resid(self, X=None) -> torch.Tensor:
        if X is None:
            X = self.X
        Y = fracdiff(X, relu(self.alpha))
        return Y[:,1:,:] - torch.bmm(X[:,:-1,:], self.A.transpose(1,2))
    
    def MSE_loss(self, X): # TODO: refactor this
        loss_fun = nn.MSELoss()
        # Y = fracdiff(self.X, relu(self.alpha))[:,1:,:]
        # Yhat = torch.bmm(self.X[:,:-1,:], self.A.transpose(1,2))
        Y = fracdiff(X, relu(self.alpha))[:,1:,:]
        Yhat = torch.bmm(X[:,:-1,:], self.A.transpose(1,2))
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
        
    def MSE(self, X=None) -> torch.Tensor:
        mse = self._MSE(X)
        return [mse[b, :self.n[b]] for b in range(self.B)]
    
    def resid(self, X=None) -> torch.Tensor:
        E = self._resid(X)
        return [E[b, :self.T[b]-1, :self.n[b]] for b in range(self.B)]
    
    def fit(self, method="OLS", **kwargs):
        if method == "OLS":
            self.fit_OLS(**kwargs)
        elif method == "LASSO":
            self.fit_LASSO(**kwargs)
        else:
            raise ValueError("`method' must be 'OLS' or 'LASSO'")
        
    def fit_LASSO(self,
                  lambda_lasso=0.01,
                  optimizer="ADAM",
                  lr=0.001,
                  n_iter=1000,
                  show_progress=False,
                  plot_losses=False):
        
        if optimizer == "ADAM":
            optimizer = Adam(self.parameters(), lr=lr)
        elif optimizer == "SGD":
            optimizer = SGD(self.parameters(), lr=lr)
        else:
            raise ValueError("`optimizer' must be 'ADAM' or 'SGD'")
        
        losses_train, losses_test = [], []
        # MSE_train, MSE_test = [], []
        MSE_train_mean, MSE_train_std = [], []
        MSE_test_mean, MSE_test_std = [], []
        X_train, X_test = self.X[:,:self.T_train,:], self.X[:,self.T_train:,:]
        for _ in tqdm(range(n_iter)) if show_progress else range(n_iter):
            optimizer.zero_grad()
            
            MSE_train = torch.hstack(self.MSE(X_train))
            MSE_train_mean.append(MSE_train.mean().item())
            MSE_train_std.append(MSE_train.std().item())
            
            MSE_test = torch.hstack(self.MSE(X_test))
            MSE_test_mean.append(MSE_test.mean().item())
            MSE_test_std.append(MSE_test.std().item())
            
            loss_test = self.MSE_loss(X_test) + lambda_lasso * self.A.abs().sum()
            loss_train = self.MSE_loss(X_train) + lambda_lasso * self.A.abs().sum()
            
            loss_train.backward()
            optimizer.step()
            
            losses_train.append(loss_train.item())
            losses_test.append(loss_test.item())
        
        if plot_losses:
            plt.plot(losses_test)
            plt.plot(losses_train)
            plt.xlabel("Iterations")
            plt.ylabel("Loss = MSE + lambda * ||A||_1")
            plt.legend(["Test", "Train"])
            plt.show()
            
            iterations = range(len(MSE_train_mean))
            
            MSE_train_mean = np.array(MSE_train_mean)
            MSE_train_std = np.array(MSE_train_std)
            
            MSE_test_mean = np.array(MSE_test_mean)
            MSE_test_std = np.array(MSE_test_std)
            
            # delta = 0.01
            delta = 0.1
            plt.plot(iterations, MSE_test_mean, label=f'Test  (final = {MSE_test_mean[-1]:.4f}, min = {MSE_test_mean.min():.4f})')
            plt.fill_between(iterations,
                            MSE_test_mean - delta*MSE_test_std,
                            MSE_test_mean + delta*MSE_test_std,
                            color='blue', alpha=0.2)
            
            plt.plot(iterations, MSE_train_mean, label=f'Train (final = {MSE_train_mean[-1]:.4f}, min = {MSE_train_mean.min():.4f})')
            plt.fill_between(iterations,
                            MSE_train_mean - delta*MSE_train_std,
                            MSE_train_mean + delta*MSE_train_std,
                            color='orange', alpha=0.2)

            
            
            plt.xlabel("Iterations")
            plt.ylabel("MSE")
            plt.legend()
            plt.show()
        
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
            
    def _forecast(self, X=None, h=1):
        if not isinstance(h, int) or h < 0:
            raise ValueError("`h' must be a non-negative integer")
        
        if X is None:
            X = self.X
    
        B, T, n = X.shape
        # forecast_kernel = -fracdiff_kernel(self.alpha, T+1)[:,1:,:]
        
        # X_hat = fftconvolve(forecast_kernel.transpose(1,2), X.transpose(1,2))[:,:,:T].transpose(1,2)
        # X_hat += torch.bmm(X, self.A.transpose(1,2))
        
        # if h == 0:
        #     return X_hat
        # else:
        #     return self._forecast(X_hat, h-1)
        
        # if h == 1:
        #     return X_hat
        # else:
        #     pass
        
        # X_hat = X.clone()    
        # for h_ in range(1, h+1):
        #     forecast_kernel = -fracdiff_kernel(self.alpha, T+h_)[:,h_:,:]
        #     X_hat = fftconvolve(forecast_kernel.transpose(1,2), X_hat.transpose(1,2))[:,:,:T].transpose(1,2) + torch.bmm(X_hat, self.A.transpose(1,2))
        # return X_hat
        
        if h == 0:
            return X
        
        X_hat = torch.zeros_like(X)
        
        fracdiff_kernel_ = fracdiff_kernel(self.alpha, h)
        for h_ in range(1, h):
            X_hat -= fracdiff_kernel_[:,h-h_,:] * self._forecast(X, h_)
        
        forecast_kernel = fracdiff_kernel(self.alpha, T+h)[:,h:,:]
        X_hat -= fftconvolve(forecast_kernel.transpose(1,2), X.transpose(1,2))[:,:,:T].transpose(1,2)
        
        X_hat += torch.bmm(self._forecast(X, h-1), self.A.transpose(1,2))
        
        return X_hat
        
    def R2(self):
        E = self.resid()
        R2 = []
        for b in range(self.B):
            X = self.X[b,:self.T[b], :self.n[b]]
            
            X_plus = X[1:,:]
            X_mean = torch.mean(X_plus, axis=0)
            
            R2_new = 1 - torch.sum(E[b]**2, axis=0) / torch.sum((X_plus - X_mean)**2, axis=0)
            R2.append(R2_new)
        
        return R2
        
    def _forecasting_resid(self, h=1):
        X = self.X
        X_hat = self._forecast(h=h)
        return X[:,h:,:] - X_hat[:,:-h,:]
    
    def forecasting_resid(self, h=1):
        E = self._forecasting_resid(h=h)
        return [E[b, :self.T[b]-h, :self.n[b]] for b in range(self.B)]
    
    def forecasting_R2(self, h=1):
        E = self.forecasting_resid(h=h)
        R2 = []
        for b in range(self.B):
            X = self.X[b,:self.T[b], :self.n[b]]
            
            X_plus = X[h:,:]
            X_mean = torch.mean(X_plus, axis=0)
            
            R2_new = 1 - torch.sum(E[b]**2, axis=0) / torch.sum((X_plus - X_mean)**2, axis=0)
            R2.append(R2_new)
        
        return R2