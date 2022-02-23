import torch
import copy
import torch.nn.utils.prune as prune
from torch.utils.data.dataloader import DataLoader
from minicifar import minicifar_train, valid_sampler
from lab1_model import ResNet, BasicBlock
from trainer import *
from utils import *
import numpy as np

GLOBAL_PRUNING = True

# Load the dataset
valid_loader = DataLoader(minicifar_train, batch_size=32, sampler=valid_sampler)

# Load the model
model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[16, 32, 64, 128], num_classes=4)
model.load_state_dict(torch.load("lab1/thinresnet18.pth"))
model = to_device(model)
loss_fn = torch.nn.CrossEntropyLoss()

amounts = []
accuracies = []
for amount in np.linspace(0, 1, 50):
    # Pruned pretrained model
    pruned_model = copy.deepcopy(model)

    if GLOBAL_PRUNING:
        # Global pruning
        parameters_to_prune = [
            (m, "weight") for m in pruned_model.modules() if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear)
        ]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    else:
        # Local pruning
        for m in pruned_model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                prune.l1_unstructured(m, name="weight", amount=amount)
    
    # Apply pruning
    for m in pruned_model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            prune.remove(m, "weight")

    # Test
    _, accuracy, class_accuracy = test_once(pruned_model, valid_loader, loss_fn)
    amount = pruning_amount(pruned_model)
    amounts.append(amount)
    accuracies.append(accuracy)

    print("amount={:.2}, accuracy={:.2f}, class_accuracy={}".format(
        amount, accuracy, [round(x, 2) for x in class_accuracy]
    ))

np.savetxt("lab3/pruning_after.csv", np.array([amounts, accuracies]).T, header="amount, accuracy")