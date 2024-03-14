import torch
from typing import Callable

# NOTE: f must be a broadcastable scalar function, i.e.
# f(x)[i_1, i_2, ..., i_n] = f(x[i_1, i_2, ..., i_n])
def bisection_optimize(f:Callable[[torch.Tensor], torch.Tensor],
                       a,
                       b,
                       max_iter,
                       tol
                       ) -> torch.Tensor:
    
    if not isinstance(a, torch.Tensor):
        raise TypeError("`a' must be a torch.Tensor")
    if not isinstance(b, torch.Tensor):
        raise TypeError("`b' must be a torch.Tensor")
    if not isinstance(max_iter, int):
        raise TypeError("`max_iter' must be an int")
    if not isinstance(tol, float):
        raise TypeError("`tol' must be a float")
    
    if a.dim() != b.dim():
        raise ValueError("`a' and `b' must have the same number of dimensions")
    if a.shape != b.shape:
        raise ValueError("`a' and `b' must have the same shape")
    if a.dtype != b.dtype:
        raise ValueError("`a' and `b' must have the same dtype")
    if ((a-b) > 0).any():
        raise ValueError("`a' must be less than or equal to `b'")
    if max_iter < 1:
        raise ValueError("`max_iter' must be at least 1")
    if tol <= 0.0:
        raise ValueError("`tol' must be positive")
    
    n_iter = 0
    while torch.max(b - a) >= tol/2 and n_iter < max_iter:
        mid = (a + b) / 2
        mid_left = torch.max(mid - tol/2, a)
        mid_right = torch.max(mid + tol/2, b)
        
        f_left = f(mid_left)
        f_right = f(mid_right)
        
        is_left_best = f_left < f_right
        b[is_left_best]  = mid[is_left_best]
        a[~is_left_best] = mid[~is_left_best]
        
        n_iter += 1
        
    x_star = (a + b) / 2
    return x_star

def bisection_grid(f:Callable[[torch.Tensor], torch.Tensor],
                   a,
                   b,
                   num_grids,
                   max_iter,
                   tol
                   ) -> torch.Tensor:
    
    if not isinstance(a, torch.Tensor):
        raise TypeError("`a' must be a torch.Tensor")
    if not isinstance(b, torch.Tensor):
        raise TypeError("`b' must be a torch.Tensor")
    if not isinstance(num_grids, int):
        raise TypeError("`num_grids' must be an int")
    if not isinstance(max_iter, int):
        raise TypeError("`max_iter' must be an int")
    if not isinstance(tol, float):
        raise TypeError("`tol' must be a float")
    
    if a.dim() != b.dim():
        raise ValueError("`a' and `b' must have the same number of dimensions")
    if a.shape != b.shape:
        raise ValueError("`a' and `b' must have the same shape")
    if a.dtype != b.dtype:
        raise ValueError("`a' and `b' must have the same dtype")
    if ((a-b) > 0).any():
        raise ValueError("`a' must be less than or equal to `b'")
    if max_iter < 1:
        raise ValueError("`max_iter' must be at least 1")
    if tol <= 0.0:
        raise ValueError("`tol' must be positive")
    
    N = num_grids
    
    global_minimizer = a
    global_minimum = f(a) + torch.inf
    for i in range(N):
        x = bisection_optimize(
            f,
            a +    i    * (b - a) / N,
            a + (i + 1) * (b - a) / N,
            max_iter,
            tol
            )
        
        fx = f(x)
        
        improved = fx < global_minimum
        global_minimizer[improved] = x[improved]
        global_minimum[improved]   = fx[improved]
    
    return global_minimizer