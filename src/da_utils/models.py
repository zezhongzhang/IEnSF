"""Simple dynamical model helpers used by experiments."""

import numpy as np
import torch


def lorenz96_drift(x, t):
    return (torch.roll(x, -1) - torch.roll(x, 2)) * torch.roll(x, 1) - x + 8


def lorenz96_drift_np(x, t):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + 8


def rk4(xt, fn, t, dt):
    k1 = fn(xt, t)
    k2 = fn(xt + dt / 2 * k1, t + dt / 2)
    k3 = fn(xt + dt / 2 * k2, t + dt / 2)
    k4 = fn(xt + dt * k3, t + dt)
    return xt + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rmse_torch(x, y):
    return torch.sqrt(torch.mean((x - y) ** 2)).item()


def rmse_np(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def state_clip(x_state, tol):
    """Replace out-of-range ensemble members with in-range copies."""
    in_range = x_state.abs().max(dim=1)[0] <= tol
    num_in_range = torch.sum(in_range).item()
    if num_in_range == 0:
        return None
    num_out_range = x_state.shape[0] - num_in_range
    pick = torch.randint(low=0, high=num_in_range, size=(num_out_range,))
    x_state_in = x_state[in_range, :]
    x_state[~in_range, :] = x_state_in[pick, :]
    return num_out_range


def state_clip_np(x_state, tol):
    return state_clip(x_state, tol)


class IntegratorODE:
    def __init__(self, drift_fun=None):
        self.drift_fun = drift_fun

    def set_drift_fun(self, drift_fun):
        self.drift_fun = drift_fun

    def integrate(self, xt, t_start, t_end, num_steps, method="fe", save_path=False):
        if self.drift_fun is None:
            raise ValueError("Drift function not specified!")
        dt = (t_end - t_start) / num_steps
        t_current = t_start
        path_all = []
        t_vec = []
        if save_path:
            path_all.append(xt)
            t_vec.append(t_current)
        for _ in range(num_steps):
            xt = self.integrate_one_step(xt, t_current, dt, method)
            t_current = t_current + dt
            if save_path:
                path_all.append(xt)
                t_vec.append(t_current)
        if save_path:
            return xt, [path_all, t_vec]
        return xt

    def integrate_one_step(self, xt, t_current, dt, method):
        if method == "rk4":
            return rk4(xt, self.drift_fun, t_current, dt)
        if method == "fe":
            return xt + self.drift_fun(xt, t_current) * dt
        raise NotImplementedError(f"Unknown ODE method: {method}")
