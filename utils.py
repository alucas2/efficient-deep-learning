import numpy as np
import matplotlib.pyplot as plt
import torch

def to_device(thing):
    if torch.cuda.is_available():
        return thing.cuda()
    else:
        return thing

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def pruning_amount(model):
    num_parameters = sum(p.numel() for p in model.parameters())
    
    # Count the number of zero in the weight mask
    num_zeros = sum((p == 0).sum().item() for (name, p) in model.named_buffers() if name.endswith("_mask"))
    
    # Count the number of parameters equal to zero if no masks are found
    if num_zeros == 0:
        num_zeros = sum((p == 0).sum().item() for p in model.parameters())
    
    return num_zeros / num_parameters
