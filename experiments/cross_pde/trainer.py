"""
Generic Parametric PINN Trainer for Cross-PDE Benchmarks
========================================================
Supports arbitrary parametric PDE residuals with PyTorch autograd.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable


class GenericParametricPINNTrainer:
    def __init__(
        self,
        model: nn.Module,
        pde_residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 3e-3,
        device: Optional[torch.device] = None
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).to(torch.float64)
        self.pde_residual_fn = pde_residual_fn
        self.inputs = inputs.to(self.device).to(torch.float64)
        self.targets = targets.to(self.device).to(torch.float64)
        self.n_samples = len(self.inputs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def compute_physics_loss(self, n_points: int = 2048) -> torch.Tensor:
        pts = min(n_points, self.n_samples)
        idx = torch.randint(0, self.n_samples, (pts,), device=self.device)
        inputs_sub = self.inputs[idx]
        inputs_grad = inputs_sub.clone().detach().requires_grad_(True)
        u_pred = self.model(inputs_grad)
        res = self.pde_residual_fn(u_pred, inputs_grad)
        return torch.mean(res ** 2)

    def train(
        self,
        epochs: int = 400,
        lbfgs_iters: int = 0,
        lambda_data: float = 1.0,
        lambda_phys: float = 0.01,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        self.model.train()
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            u_pred = self.model(self.inputs)
            l_data = torch.mean((u_pred - self.targets) ** 2)
            if lambda_phys > 0.0:
                l_phys = self.compute_physics_loss(n_points=2048)
            else:
                l_phys = torch.tensor(0.0, device=self.device, dtype=torch.float64)
            loss = lambda_data * l_data + lambda_phys * l_phys
            loss.backward()
            self.optimizer.step()

        if lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(self.model.parameters(), max_iter=lbfgs_iters, tolerance_grad=1e-12, tolerance_change=1e-12)
            def closure():
                lbfgs.zero_grad()
                u_p = self.model(self.inputs)
                ld = torch.mean((u_p - self.targets) ** 2)
                lp = self.compute_physics_loss(n_points=2048) if lambda_phys > 0.0 else torch.tensor(0.0, device=self.device, dtype=torch.float64)
                ls = lambda_data * ld + lambda_phys * lp
                ls.backward()
                return ls
            lbfgs.step(closure)

        self.model.eval()
        return {"training_time_s": time.time() - start_time}
