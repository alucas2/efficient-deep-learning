import torch

def set_binarized_weights(module, _input):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        module.weight.data = torch.nn.functional.hardtanh(module.weight.data, min_val=-1.0, max_val=1.0) # Clip
        module.weight_orig = module.weight.clone() # Save clipped weights
        module.weight.data = 2.0 * (module.weight.data >= 0) - 1.0 # Binarize

def set_original_weights(module, _grad_input, _grad_output):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        module.weight.data = module.weight_orig.data # Restore original weights
        del module.weight_orig

class QuantizedWeight(torch.nn.Module):
    """Wrapper for a quantized module"""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.register_forward_pre_hook(set_binarized_weights)
        self.register_backward_hook(set_original_weights)

    def forward(self, x):
        return self.module(x)

class QuantizedActivation(torch.nn.Module):
    pass