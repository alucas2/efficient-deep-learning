import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, target):
        log_softmax = F.log_softmax(y, dim=1)
        return -torch.sum(log_softmax * target) / y.size(0)

class DistillationLoss(torch.nn.Module):
    def __init__(self, teacher_model, lambda_param):
        super().__init__()
        self.lambda_param = lambda_param
        self.div_loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.teacher_model = teacher_model

    def forward(self, x, y, target):
        student_log_softmax = F.log_softmax(y, dim=1)
        teacher_softmax = F.softmax(self.teacher_model(x), dim=1)
        div_loss = self.div_loss(student_log_softmax, teacher_softmax)
        normal_loss = -torch.sum(student_log_softmax * target) / y.size(0)
        return normal_loss + self.lambda_param * div_loss
        

def to_device(thing):
    if torch.cuda.is_available():
        return thing.cuda()
    else:
        return thing

def to_same_device(thing, dest_device):
    if dest_device.get_device() != -1:
        return thing.to(dest_device.get_device())
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
