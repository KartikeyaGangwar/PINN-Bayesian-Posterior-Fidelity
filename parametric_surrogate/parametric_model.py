r"""
Parametric Neural Network Model Module (1D Heat Equation)
==========================================================
Architectural Details:
    - 3D Inputs: (x, t, \alpha) \in [0, 1] \times [0, 1] \times [\alpha_{\min}, \alpha_{\max}]
    - Output: u_{\text{PINN}}(x, t; \alpha)
    - Fourier Feature Mapping for coordinate encoding
    - Modified MLP with Multiplicative Coordinate Gating (U, V encoders)
    - Double precision (torch.float64) throughout
"""

import numpy as np
import torch
import torch.nn as nn


class ParametricFourierFeatures(nn.Module):
    """Fourier Feature Mapping for 3D input coordinates (x, t, alpha)."""
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
    Parametric Modified MLP Architecture for 1D Heat Equation.
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
