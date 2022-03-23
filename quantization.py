import torch
import numpy as np

def quantize(x, steps):
    x = np.clip(x, -1.0, 1.0)
    x = np.round(x * steps) / steps

def set_binarized_weights(module, _input):
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            m.weight.data = torch.nn.functional.hardtanh(m.weight.data, min_val=-1.0, max_val=1.0) # Clip
            m.weight_orig = m.weight.clone() # Save clipped weights
            m.weight.data = 2.0 * (m.weight.data >= 0) - 1.0 # Binarize

def set_original_weights(module, _grad_input, _grad_output):
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            m.weight.data = m.weight_orig.data # Restore original weights
            del m.weight_orig

class Quantized(torch.nn.Module):
    """Wrapper for a quantized module"""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.register_forward_pre_hook(set_binarized_weights)
        self.register_backward_hook(set_original_weights)

    def forward(self, x):
        # Quantize activation
        x = 2.0 * (x >= 0) - 1.0
        return self.module(x)
