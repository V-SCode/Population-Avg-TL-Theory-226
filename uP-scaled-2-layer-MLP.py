import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MuPMLP(nn.Module):
    """
    Two-layer MLP with mean-field / µP scaling:

        h_i(x)   = (1/sqrt(D)) * W_i · x
        phi(h)   = linear or ReLU
        f(x)     = (1 / (gamma0 * N)) * sum_i w_i * phi(h_i(x))

    Parameters are:
        W: (N, D)
        w: (N,)
    """
    def __init__(self, D: int, N: int, gamma0: float = 1.0, activation: str = "linear", device=None):
        super().__init__()
        self.D = D
        self.N = N
        self.gamma0 = gamma0
        self.activation_type = activation

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.W = nn.Parameter(torch.randn(N, D, device=device))

        self.w = nn.Parameter(torch.randn(N, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, D)
        returns: (batch_size,) predictions
        """
        x = x.to(self.device).float()

        h = (1.0 / math.sqrt(self.D)) * x @ self.W.t()

        if self.activation_type == "relu":
            phi = torch.relu(h)
        elif self.activation_type == "linear":
            phi = h
        else:
            raise ValueError(f"Unknown activation {self.activation_type}")

        out = (1.0 / (self.gamma0 * self.N)) * (phi @ self.w)
        return out

