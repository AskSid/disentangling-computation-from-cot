import torch
import numpy as np

import random as _random

def set_seed(seed: int):
    """Set random seeds for reproducibility across python, numpy, and torch."""
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False