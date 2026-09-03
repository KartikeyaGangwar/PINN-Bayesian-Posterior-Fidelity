r"""
PyTorch Dataset Module for 3D Parametric Inputs (1D Heat Equation)
===================================================================
Inputs: (x, t, \alpha) \in [0, 1] \times [0, 1] \times \Theta
Targets: u(x, t; \alpha)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any


class ParametricPINNDataset(Dataset):
    r"""
    PyTorch Dataset wrapping 3D inputs (x, t, \alpha) and target analytical field values u.
    """
    def __init__(self, dataset_dict: Dict[str, np.ndarray]):
        X_grid = dataset_dict["X_grid"]
        T_grid = dataset_dict["T_grid"]
        alphas = dataset_dict["alpha_values"]
        u_fields = dataset_dict["u_fields"]
        
        n_params, nx, nt = u_fields.shape
        n_points_per_param = nx * nt
        n_total = n_params * n_points_per_param
        
        inputs = np.zeros((n_total, 3), dtype=np.float64)
        targets = np.zeros((n_total, 1), dtype=np.float64)
        
        idx = 0
        X_flat = X_grid.ravel()
        T_flat = T_grid.ravel()
        
        for i, a_val in enumerate(alphas):
            u_flat = u_fields[i].ravel()
            
            inputs[idx:idx + n_points_per_param, 0] = X_flat
            inputs[idx:idx + n_points_per_param, 1] = T_flat
            inputs[idx:idx + n_points_per_param, 2] = a_val
            
            targets[idx:idx + n_points_per_param, 0] = u_flat
            idx += n_points_per_param
            
        self.inputs = torch.tensor(inputs, dtype=torch.float64)
        self.targets = torch.tensor(targets, dtype=torch.float64)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]
