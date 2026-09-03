r"""
Parametric PINN Trainer Module (1D Heat Equation)
=================================================
GPU-resident vectorized trainer for 1D Parametric Heat Equation.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .dataset import ParametricPINNDataset


class ParametricPINNTrainer:
    """
    High-performance GPU-resident Trainer for Parametric PINN Surrogate.
    """
    def __init__(
        self,
        model: nn.Module,
        dataset: ParametricPINNDataset,
        learning_rate: float = 3e-3,
        batch_size: int = 16384,
        device: Optional[torch.device] = None
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).to(torch.float64)
        
        # Keep entire dataset resident in GPU VRAM (very small, < 10 MB)
        self.inputs = dataset.inputs.to(self.device)
        self.targets = dataset.targets.to(self.device)
        self.n_samples = len(self.inputs)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def compute_physics_loss(self, inputs: Optional[torch.Tensor] = None, n_points: int = 2048) -> torch.Tensor:
        """Computes parametric PDE residual f(x,t; alpha) = u_t - alpha * u_xx."""
        if isinstance(inputs, torch.Tensor):
            inputs_sub = inputs.to(self.device)
        else:
            pts = n_points if isinstance(inputs, (int, type(None))) and not isinstance(inputs, int) else (inputs if isinstance(inputs, int) else n_points)
            pts = min(pts, self.n_samples)
            idx = torch.randint(0, self.n_samples, (pts,), device=self.device)
            inputs_sub = self.inputs[idx]

        inputs_grad = inputs_sub.clone().detach().requires_grad_(True)
        u_pred = self.model(inputs_grad)
        
        grads = torch.autograd.grad(
            u_pred, inputs_grad,
            torch.ones_like(u_pred),
            create_graph=True
        )[0]
        
        dudx = grads[:, 0:1]
        dudt = grads[:, 1:2]
        alpha_val = inputs_grad[:, 2:3]
        
        d2udx2 = torch.autograd.grad(
            dudx, inputs_grad,
            torch.ones_like(dudx),
            create_graph=True
        )[0][:, 0:1]
        
        f = dudt - alpha_val * d2udx2
        return torch.mean(f**2)

    def train(
        self,
        epochs: int = 1000,
        lbfgs_iters: int = 200,
        lambda_data: float = 1.0,
        lambda_phys: float = 0.01,
        log_interval: int = 200
    ) -> Dict[str, Any]:
        """
        Executes Adam optimization followed by L-BFGS refinement.
        """
        print(f"\n--- Starting Parametric PINN Training ({epochs} Adam epochs) ---", flush=True)
        start_time = time.time()
        
        self.model.train()
        
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            
            u_pred = self.model(self.inputs)
            L_data = torch.mean((u_pred - self.targets)**2)
            
            if lambda_phys > 0.0:
                L_phys = self.compute_physics_loss(n_points=2048)
            else:
                L_phys = torch.tensor(0.0, device=self.device)
                
            total_loss = lambda_data * L_data + lambda_phys * L_phys
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % log_interval == 0 or epoch == epochs:
                print(
                    f"  Epoch {epoch:4d}/{epochs} | "
                    f"Total: {total_loss.item():.4e} | "
                    f"Data: {L_data.item():.4e} | "
                    f"Phys: {L_phys.item():.4e}",
                    flush=True
                )

        # L-BFGS Refinement
        if lbfgs_iters > 0:
            print(f"\n--- Refining with L-BFGS ({lbfgs_iters} max iterations) ---", flush=True)
            lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                max_iter=lbfgs_iters,
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
                history_size=30,
                line_search_fn="strong_wolfe"
            )
            
            def closure():
                lbfgs.zero_grad()
                pred = self.model(self.inputs)
                l_data = torch.mean((pred - self.targets)**2)
                if lambda_phys > 0.0:
                    l_phys = self.compute_physics_loss(n_points=2048)
                else:
                    l_phys = torch.tensor(0.0, device=self.device)
                tot = lambda_data * l_data + lambda_phys * l_phys
                tot.backward()
                return tot
                
            lbfgs.step(closure)
            print("  L-BFGS refinement completed.", flush=True)

        elapsed = time.time() - start_time
        print(f"--- Training Completed in {elapsed:.2f}s ---\n", flush=True)
        
        self.model.eval()
        return {"training_time_seconds": elapsed}
