"""Result recording helpers for DA experiments."""

import numpy as np
import torch


class ResultRecorder:
    def __init__(self, rmse=True, crps=False, spread=False, record_est=False, record_true_state=False, obs_index=None):
        """
        Initialize the ResultRecorder.
        
        Args:
            rmse: Whether to compute RMSE
            crps: Whether to compute CRPS
            spread: Whether to compute ensemble spread
            record_est: Whether to record ensemble estimates
            record_true_state: Whether to record true states
            obs_index: Boolean tensor (dim_x) indicating observed dimensions
        """
        self.record_rmse = rmse
        self.record_spread = spread
        self.record_crps = crps
        self.record_est = record_est
        self.record_true_state = record_true_state
        self.obs_index = obs_index
        
        # Initialize storage dictionaries
        self.results = {
            'time': [],
            'posterior': []
        }
        
        if self.record_rmse:
            self.results['rmse'] = []
            if obs_index is not None:
                self.results['rmse_obs'] = []
                self.results['rmse_non_obs'] = []
        
        if self.record_crps:
            self.results['crps'] = []
            if obs_index is not None:
                self.results['crps_obs'] = []
                self.results['crps_non_obs'] = []
        
        if self.record_spread:
            self.results['spread'] = []
            if obs_index is not None:
                self.results['spread_obs'] = []
                self.results['spread_non_obs'] = []
        
        if self.record_est:
            self.results['ensemble_mean'] = []
            if obs_index is not None:
                self.results['ensemble_mean_obs'] = []
                self.results['ensemble_mean_non_obs'] = []
        
        if self.record_true_state:
            self.results['true_state'] = []
            if obs_index is not None:
                self.results['true_state_obs'] = []
                self.results['true_state_non_obs'] = []
    
    def record(self, state_true, ensemble, t, posterior=False):
        """
        Record results at a given time step.
        
        Args:
            state_true: Tensor (dim_x) - true state
            ensemble: Tensor (ensemble_size, dim_x) - ensemble members
            t: scalar - time step
            posterior: bool - whether this is posterior (True) or prior (False)
        """
        # Record time and posterior indicator
        self.results['time'].append(t)
        self.results['posterior'].append(posterior)
        
        # Compute and record RMSE
        if self.record_rmse:
            rmse_full = self.rmse(state_true, ensemble)
            self.results['rmse'].append(rmse_full.item())
            
            if self.obs_index is not None:
                rmse_obs = self.rmse(state_true[self.obs_index], ensemble[:, self.obs_index])
                rmse_non_obs = self.rmse(state_true[~self.obs_index], ensemble[:, ~self.obs_index])
                self.results['rmse_obs'].append(rmse_obs.item())
                self.results['rmse_non_obs'].append(rmse_non_obs.item())
        
        # Compute and record CRPS
        if self.record_crps:
            crps_full = self.crps(state_true, ensemble)
            self.results['crps'].append(crps_full.item())
            
            if self.obs_index is not None:
                crps_obs = self.crps(state_true[self.obs_index], ensemble[:, self.obs_index])
                crps_non_obs = self.crps(state_true[~self.obs_index], ensemble[:, ~self.obs_index])
                self.results['crps_obs'].append(crps_obs.item())
                self.results['crps_non_obs'].append(crps_non_obs.item())
        
        # Compute and record spread
        if self.record_spread:
            spread_full = self.spread(ensemble)
            self.results['spread'].append(spread_full.item())
            
            if self.obs_index is not None:
                spread_obs = self.spread(ensemble[:, self.obs_index])
                spread_non_obs = self.spread(ensemble[:, ~self.obs_index])
                self.results['spread_obs'].append(spread_obs.item())
                self.results['spread_non_obs'].append(spread_non_obs.item())
        
        # Record ensemble mean if requested
        if self.record_est:
            ens_mean = ensemble.mean(dim=0)
            self.results['ensemble_mean'].append(ens_mean.clone())
            
            if self.obs_index is not None:
                self.results['ensemble_mean_obs'].append(ens_mean[self.obs_index].clone())
                self.results['ensemble_mean_non_obs'].append(ens_mean[~self.obs_index].clone())
        
        # Record true state if requested
        if self.record_true_state:
            self.results['true_state'].append(state_true.clone())
            
            if self.obs_index is not None:
                self.results['true_state_obs'].append(state_true[self.obs_index].clone())
                self.results['true_state_non_obs'].append(state_true[~self.obs_index].clone())
    
    @staticmethod
    def rmse(state_true, ensemble):
        """
        Compute Root Mean Square Error.
        
        Args:
            state_true: Tensor (..., dim_x) - true state
            ensemble: Tensor (ensemble_size, ..., dim_x) - ensemble members
        
        Returns:
            RMSE: Tensor (scalar)
        """
        ensemble_mean = ensemble.mean(dim=0)
        mse = torch.mean((ensemble_mean - state_true) ** 2)
        return torch.sqrt(mse)
    
    @staticmethod
    def crps(state_true, ensemble):
        """
        Compute Continuous Ranked Probability Score.
        
        Args:
            state_true: Tensor (..., dim_x) - true state
            ensemble: Tensor (ensemble_size, ..., dim_x) - ensemble members
        
        Returns:
            CRPS: Tensor (scalar)
        """
        ensemble_size = ensemble.shape[0]
        
        # First term: mean absolute error between ensemble and truth
        term1 = torch.mean(torch.abs(ensemble - state_true))
        
        # Second term: mean pairwise distance within ensemble
        # Compute all pairwise differences
        diff = ensemble.unsqueeze(0) - ensemble.unsqueeze(1)  # (ens_size, ens_size, dim_x)
        term2 = torch.mean(torch.abs(diff)) / 2
        
        crps = term1 - term2
        return crps
    
    @staticmethod
    def spread(ensemble):
        """
        Compute ensemble spread (standard deviation).
        
        Args:
            ensemble: Tensor (ensemble_size, ..., dim_x) - ensemble members
        
        Returns:
            Spread: Tensor (scalar) - mean standard deviation across all dimensions
        """
        return torch.mean(torch.std(ensemble, dim=0))
    
    def get_results(self):
        """Return the recorded results dictionary."""
        return self.results
    
    def clear(self):
        """Clear all recorded results."""
        for key in self.results:
            self.results[key] = []
            
    def to_numpy(self):
        """
        Convert all results to numpy arrays.
        
        Returns:
            dict: Dictionary with all results converted to numpy arrays
        """
        
        numpy_results = {}
        for key, value in self.results.items():
            if len(value) == 0:
                numpy_results[key] = np.array([])
            elif isinstance(value[0], torch.Tensor):
                # Stack tensors and convert to numpy
                numpy_results[key] = torch.stack(value).cpu().numpy()
            elif isinstance(value[0], bool):
                # Convert list of bools to numpy array
                numpy_results[key] = np.array(value, dtype=bool)
            else:
                # Convert list of scalars to numpy array
                numpy_results[key] = np.array(value)
        
        return numpy_results