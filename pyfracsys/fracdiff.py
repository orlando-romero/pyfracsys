import torch
from torchaudio.functional import fftconvolve

# TODO: implement in-place fracdiff_kernel
def fracdiff_kernel_(Psi, alpha, T):
    pass

def fracdiff_kernel(alpha: torch.Tensor, T: int) -> torch.Tensor:
    if not isinstance(alpha, torch.Tensor):
        raise TypeError("alpha must be a torch.Tensor")
    if not isinstance(T, int):
        raise TypeError("T must be an int")
    
    if alpha.dim() != 2:
        raise TypeError("alpha must be a 2d array")
    if (alpha < 0).any():
        raise ValueError("Every value in alpha must be non-negative")
    
    if T < 1:
        raise ValueError("T must be at least 1")
        
    # TODO refactor this!!!
    
    B, nn = alpha.shape
    alpha = alpha.flatten()
    
    n = len(alpha)
    
    t_range = torch.arange(1, T).to(alpha.device, dtype=alpha.dtype).view(-1, 1)  # Reshape for broadcasting
    alpha_expanded = alpha.expand(T-1, n)  # Expand alpha for broadcasting
    numerator = -(alpha_expanded - t_range + 1)
    denominator = t_range
 
    Psi_values = torch.cumprod(numerator / denominator, dim=0)
    Psi = torch.cat([torch.ones(1, n).to(alpha.device, dtype=alpha.dtype), Psi_values], dim=0)

    alpha = alpha.reshape(B, nn)
    Psi = Psi.reshape(B, T, nn)
    
    return Psi

def fracdiff(X: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor")
    if not isinstance(alpha, torch.Tensor):
        raise TypeError("alpha must be a torch.Tensor")
    if X.dim() != 3:
        raise TypeError("X must be a 3D array")
    if alpha.dim() != 2:
        raise TypeError("alpha must be a 2d array")
    
    if X.device != alpha.device:
        raise ValueError("X and alpha must be on the same device")
    if X.dtype != alpha.dtype:
        raise ValueError("X and alpha must have the same precision")
    
    B, T, n = X.shape
    if alpha.shape != (B, n):
        raise TypeError("alpha must be a (B,n) array mattching the (B,T,n) array X")
    
    Psi = fracdiff_kernel(alpha, T)
    
    
    return fftconvolve(Psi.transpose(1, 2), X.transpose(1,2))[:,:,:T].transpose(1,2)