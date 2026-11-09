
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = Path(__file__).parent.resolve()

# DA problem parameters 
dtype = torch.float32
TOTAL_STEPS = 100
STATE_SIGMA = 0.5
OBS_SIGMA = 0.5
RNG_SEED = 42
DIM_STATE = 2
DIM_OBS = 1


# Define the linear Gaussian model (discrete-time 2D oscillator)
omega = 2.0
dt = 0.1
wdt = omega  * dt
c = torch.cos(torch.tensor(wdt, dtype=dtype))
s = torch.sin(torch.tensor(wdt, dtype=dtype))


A = torch.tensor([[c, s/omega],
                 [-omega*s, c]], dtype=dtype)
H = torch.tensor([[1.0, 0.0]], dtype=dtype) 

Q = torch.eye(DIM_STATE, dtype=dtype) * STATE_SIGMA**2

R = torch.eye(DIM_OBS, dtype=dtype) * OBS_SIGMA**2

INIT_STATE = torch.tensor([[3.0, -3.0]], dtype=dtype)  # (1, DIM_STATE)

def step_forward(x, A, Q, noisy=True):
    """
    One-step forward simulation for a linear state-space model.
    
    Args:
        A (torch.Tensor): State transition matrix (n x n).
        Q (torch.Tensor): Process noise covariance (n x n).
        x (torch.Tensor): Current state(s), shape (..., n).
                          Can be a single state (n,) or batch (batch_size, n).
        noisy (bool): If True, adds Gaussian process noise.
    
    Returns:
        torch.Tensor: Next state(s), same shape as x.
    """
    # Ensure Q is valid
    n = A.shape[0]
    L = torch.linalg.cholesky(Q)  # for sampling correlated noise

    # Linear propagation
    x_next = x @ A.T  # works with batches (matrix multiplication)

    if noisy:
        if x.dim() == 1:  # single state
            eps = torch.randn(n, dtype=x.dtype, device=x.device)
            noise = eps @ L.T
        else:  # batch of states
            batch_size = x.shape[0]
            eps = torch.randn(batch_size, n, dtype=x.dtype, device=x.device)
            noise = eps @ L.T
        x_next = x_next + noise

    return x_next



if __name__ == "__main__":

    x = INIT_STATE.clone()
    state_path = [x.cpu().numpy()]
    for i in range(TOTAL_STEPS):
        x = step_forward(x, A, Q, noisy=True)
        state_path.append(x.cpu().numpy())
    state_path = np.vstack(state_path)  # (TOTAL_STEPS+1, DIM_STATE)

    obs_path = (state_path @ H.T.numpy()) + np.random.randn(TOTAL_STEPS+1, DIM_OBS) * OBS_SIGMA  # (TOTAL_STEPS+1, DIM_OBS)
    
    # save the problem data
    folder = BASE_DIR / "problem_data"
    os.makedirs(folder, exist_ok=True)
    np.save(folder / "obs_path.npy", obs_path)
    np.save(folder / "state_path.npy", state_path)
    
    # plot the path
    plt.plot(state_path[:, 0],'b-', label='True position 1')
    plt.plot(state_path[:, 1],'b-', label='True position 2')
    plt.plot(obs_path[:, 0],'r.-', label='obs 1')
    plt.legend()
    plt.show()

    # plot the path in 2D
    plt.plot(*state_path.T, '.-' , color='k')
    plt.show()