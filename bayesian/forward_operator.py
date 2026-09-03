r"""
Forward Operator & Parametric PINN Surrogate Module (1D Heat Equation)
======================================================================
Mathematical Formulation:
    Governing PDE:
        u_t = \alpha u_{xx}, \quad x \in [0, 1], \quad t \in [0, 1]
    Boundary Conditions:
        u(0, t) = 0, \quad u(1, t) = 0
    Initial Condition:
        u(x, 0) = \sin(\pi x)

    Exact Analytical Solution:
        u_{\text{exact}}(x, t; \alpha) = \exp(-\alpha \pi^2 t) \sin(\pi x)

    Parametric Forward Map:
        \mathcal{F}: \alpha \mapsto u(x, t; \alpha)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional, Any, Callable, Union


# =============================================================================
# FORWARD SOLVER OUTPUT CONTAINER
# =============================================================================

@dataclass
class ForwardSolverOutput:
    """Structured container returned by forward operators."""
    u_exact: Optional[np.ndarray] = None
    u_pinn: Optional[np.ndarray] = None
    alpha: float = 0.5

    @property
    def u(self) -> np.ndarray:
        """Returns the primary solution field produced by this operator."""
        if self.u_pinn is not None:
            return self.u_pinn
        return self.u_exact


# =============================================================================
# ABSTRACT FORWARD OPERATOR BASE CLASS
# =============================================================================

class BaseForwardOperator:
    """Abstract Base Class for solver-agnostic Bayesian forward operators."""
    def evaluate(self, alpha: Union[float, np.ndarray]) -> ForwardSolverOutput:
        raise NotImplementedError
        
    def __call__(self, alpha: Union[float, np.ndarray]) -> ForwardSolverOutput:
        return self.evaluate(alpha)


# =============================================================================
# 1. EXACT ANALYTICAL FORWARD OPERATOR
# =============================================================================

class ExactForwardOperator(BaseForwardOperator):
    r"""
    Evaluates the exact analytical ground-truth solution:
        u_{\text{exact}}(x, t; \alpha) = \exp(-\alpha \pi^2 t) \sin(\pi x)
    """
    def __init__(self, resolution: int = 100):
        self.resolution = resolution
        self.x_1d = np.linspace(0.0, 1.0, resolution)
        self.t_1d = np.linspace(0.0, 1.0, resolution)
        self.X_grid, self.T_grid = np.meshgrid(self.x_1d, self.t_1d, indexing="ij")

    def evaluate(self, alpha: Union[float, np.ndarray]) -> ForwardSolverOutput:
        if isinstance(alpha, (list, tuple, np.ndarray)):
            alpha_val = float(alpha[0])
        else:
            alpha_val = float(alpha)

        u_exact = np.exp(-alpha_val * (np.pi ** 2) * self.T_grid) * np.sin(np.pi * self.X_grid)
        return ForwardSolverOutput(u_exact=u_exact, u_pinn=None, alpha=alpha_val)


# =============================================================================
# 2. PARAMETRIC PINN NEURAL NETWORK ARCHITECTURE
# =============================================================================

class ParametricFourierFeatures(nn.Module):
    """Fourier feature mapping for 3D input coordinates (x, t, alpha)."""
    def __init__(self, in_features: int = 3, out_features: int = 64, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.B = nn.Parameter(
            torch.randn(in_features, out_features // 2, dtype=torch.float64) * scale,
            requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(xp), torch.cos(xp)], dim=-1)


class ParametricModifiedMLP(nn.Module):
    r"""
    Parametric Modified MLP for 1D Heat Equation.
    3D Inputs: (x, t, \alpha) \mapsto u(x, t; \alpha).
    """
    def __init__(
        self,
        n_input: int = 3,
        n_output: int = 1,
        n_hidden: int = 64,
        n_layers: int = 4,
        use_fourier: bool = False,
        fourier_scale: float = 1.0
    ):
        super().__init__()
        self.use_fourier = use_fourier
        self.activation = nn.Tanh()
        
        in_dim = n_hidden if use_fourier else n_input
        if use_fourier:
            self.fourier = ParametricFourierFeatures(n_input, n_hidden, scale=fourier_scale)
            
        self.encoder_U = nn.Linear(in_dim, n_hidden, dtype=torch.float64)
        self.encoder_V = nn.Linear(in_dim, n_hidden, dtype=torch.float64)
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_hidden, n_hidden, dtype=torch.float64) for _ in range(n_layers)
        ])
        self.out_layer = nn.Linear(n_hidden, n_output, dtype=torch.float64)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            x_feat = self.fourier(x)
        else:
            x_feat = x
            
        U = self.activation(self.encoder_U(x_feat))
        V = self.activation(self.encoder_V(x_feat))
        
        H = U
        for layer in self.hidden_layers:
            Z = self.activation(layer(H))
            H = (1.0 - Z) * U + Z * V
            
        return self.out_layer(H)


# =============================================================================
# 3. PARAMETRIC PINN FORWARD OPERATOR
# =============================================================================

class ParametricPINNForwardOperator(BaseForwardOperator):
    r"""
    Parametric PINN Surrogate Forward Operator.
    Evaluates trained neural surrogate network u_{\text{PINN}}(x, t; \alpha).
    """
    def __init__(
        self,
        model: nn.Module,
        resolution: int = 100,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.resolution = resolution
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).to(torch.float64).eval()
        
        x_grid = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64, device=self.device)
        t_grid = torch.linspace(0.0, 1.0, resolution, dtype=torch.float64, device=self.device)
        self.X_grid, self.T_grid = torch.meshgrid(x_grid, t_grid, indexing="ij")
        self.X_flat = self.X_grid.reshape(-1, 1)
        self.T_flat = self.T_grid.reshape(-1, 1)

    def evaluate(self, alpha: Union[float, np.ndarray]) -> ForwardSolverOutput:
        if isinstance(alpha, (list, tuple, np.ndarray)):
            alpha_val = float(alpha[0])
        else:
            alpha_val = float(alpha)

        alpha_tensor = torch.full_like(self.X_flat, alpha_val, dtype=torch.float64, device=self.device)
        inputs = torch.cat([self.X_flat, self.T_flat, alpha_tensor], dim=1)
        
        with torch.no_grad():
            u_pred = self.model(inputs).reshape(self.resolution, self.resolution).cpu().numpy()
            
        # Ground truth analytical for instantaneous error tracking
        X_np = self.X_grid.cpu().numpy()
        T_np = self.T_grid.cpu().numpy()
        u_exact = np.exp(-alpha_val * (np.pi ** 2) * T_np) * np.sin(np.pi * X_np)
        
        return ForwardSolverOutput(u_exact=u_exact, u_pinn=u_pred, alpha=alpha_val)
