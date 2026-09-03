r"""
Observation Operator Module (\mathcal{H})
=======================================
Mathematical Reference:
    Alexanderian (2021) "Optimal Experimental Design for Infinite-Dimensional Bayesian Inverse Problems"
    Stuart (2010) "Inverse problems: A Bayesian perspective", Acta Numerica

Mathematical Equation:
    \boldsymbol{y}_{\text{pred}} = \mathcal{H}(u)
    
Description:
    The Observation Operator \mathcal{H} maps a continuous or discretized forward PDE solution 
    field u(x,t) \in \mathcal{V} to a finite-dimensional measurement space \mathbb{R}^{M}.
    
    This operator is completely decoupled from the forward PDE solver (PINN) and Bayesian likelihood.
    It supports:
    - Sparse point sensors (sensor coordinates (x_i, t_i))
    - Dense spatial-temporal grids
    - Partial observations (e.g. final-time measurements u(x, T))
    - Linear/Nonlinear spatial averaging operators
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union


class ObservationOperator:
    r"""
    Observation Operator \mathcal{H}: \mathcal{V} \to \mathbb{R}^M
    Extracts simulated measurement predictions \boldsymbol{y}_{\text{pred}} from forward solution u(x,t).
    """
    def __init__(
        self, 
        sensor_locations: Optional[np.ndarray] = None, 
        grid_shape: Optional[Tuple[int, int]] = None
    ):
        """
        Inputs:
            sensor_locations: Array of sensor coordinates [M, 2] containing (x, t) pairs.
                              If None, extracts all grid points.
            grid_shape: Grid dimensions (N_x, N_t) if u is passed as a 2D mesh matrix.
        """
        self.sensor_locations = sensor_locations
        self.grid_shape = grid_shape

    def observe(self, u_field: Union[np.ndarray, torch.Tensor], x_grid: Optional[np.ndarray] = None, t_grid: Optional[np.ndarray] = None) -> np.ndarray:
        r"""
        Applies observation operator \boldsymbol{y}_{\text{pred}} = \mathcal{H}(u).
        
        Mathematical Formulation:
            \mathcal{H}(u)_m = u(x_m, t_m), \quad m = 1, \dots, M
            
        Inputs:
            u_field: Solution field matrix [N_x, N_t] or flat array [N_x * N_t]
            x_grid: 1D spatial grid coordinates [N_x] (optional, used for interpolation)
            t_grid: 1D temporal grid coordinates [N_t] (optional, used for interpolation)
            
        Outputs:
            y_pred: Extracted measurement vector [M]
        """
        if isinstance(u_field, torch.Tensor):
            u_arr = u_field.detach().cpu().numpy()
        else:
            u_arr = np.array(u_field)

        # Case 1: Sparse point sensors evaluated at specific (x,t) coordinates
        if self.sensor_locations is not None:
            if u_arr.ndim == 2 and x_grid is not None and t_grid is not None:
                # Interpolate from 2D grid onto sensor points
                from scipy.interpolate import RegularGridInterpolator
                interp = RegularGridInterpolator((x_grid, t_grid), u_arr, bounds_error=False, fill_value=None)
                return interp(self.sensor_locations)
            elif u_arr.ndim == 1:
                # Direct indexing if u_field corresponds to sensor points
                return u_arr.flatten()
            else:
                # Bilinear or nearest sampling from 2D grid assuming [0,1] domain
                N_x, N_t = u_arr.shape
                x_pts = self.sensor_locations[:, 0]
                t_pts = self.sensor_locations[:, 1]
                idx_x = np.clip(np.round(x_pts * (N_x - 1)).astype(int), 0, N_x - 1)
                idx_t = np.clip(np.round(t_pts * (N_t - 1)).astype(int), 0, N_t - 1)
                return u_arr[idx_x, idx_t]
        
        # Case 2: Dense full-field observation
        return u_arr.flatten()

    def __call__(self, u_field: Union[np.ndarray, torch.Tensor], x_grid: Optional[np.ndarray] = None, t_grid: Optional[np.ndarray] = None) -> np.ndarray:
        return self.observe(u_field, x_grid, t_grid)
