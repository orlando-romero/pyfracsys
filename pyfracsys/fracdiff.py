import torch
from torchaudio.functional import fftconvolve

# TODO: implement in-place fracdiff_kernel
def fracdiff_kernel_(B, alpha, T):
    pass

def fracdiff_kernel(alpha: torch.Tensor, T: int) -> torch.Tensor:
    if not isinstance(alpha, torch.Tensor):
        raise TypeError("alpha must be a torch.Tensor")
    if not isinstance(T, int):
        raise TypeError("T must be an int")
    if alpha.dim() != 0 and alpha.dim() != 1:
        raise TypeError("alpha must be a scalar or a 1d array")
    if (alpha < 0).any():
        raise ValueError("Every value in alpha must be non-negative")
    if T < 1:
        raise ValueError("T must be at least 1")
    
    if alpha.dim() == 0:
        alpha = alpha.unsqueeze(0)
        
    n = len(alpha)
    B = torch.zeros(T, n).to(alpha.device, dtype=alpha.dtype)
    for i in range(n):
        t_range = torch.arange(T-1).to(alpha.device, dtype=alpha.dtype)
        B[:, i] = torch.cat([torch.tensor([1.0]).to(alpha.device, dtype=alpha.dtype), torch.cumprod(-(alpha[i] - t_range) / (t_range + 1), dim=0)])
      
    return B

def fracdiff(X: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor")
    if not isinstance(alpha, torch.Tensor):
        raise TypeError("alpha must be a torch.Tensor")
    if X.dim() != 2:
        raise TypeError("X must be a 2D array")
    
    if X.device != alpha.device:
        raise ValueError("X and alpha must be on the same device")
    if X.dtype != alpha.dtype:
        raise ValueError("X and alpha must have the same precision")
    
    T, n = X.shape
    if T < n:
        raise TypeError("X must be T x n array with T >= n") # Do not check here!
    if alpha.dim() != 0 and alpha.dim() != 1:
        raise TypeError("alpha must be a scalar or a 1d array")
    if alpha.dim() == 1 and len(alpha) != n:
        raise TypeError("If alpha is a 1d array, then it must match the number of channels in the data")
    
    B = fracdiff_kernel(alpha, T)

    return fftconvolve(B.T, X.T)[:, :T].T