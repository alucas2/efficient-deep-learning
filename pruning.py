from matplotlib.pyplot import locator_params
import torch
import torch.nn.utils.prune as prune

class Pruned(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.prune_unstructured(0.0) # create all the "weights_orig" attributes

    def prune_unstructured(self, amount):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                prune.l1_unstructured(m, name="weight", amount=amount)

    def forward(self, x):
        return self.model(x)
