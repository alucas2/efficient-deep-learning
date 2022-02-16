import torch
import copy
import torch.nn.utils.prune as prune
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from minicifar import minicifar_test
from lab1_model import ResNet, BasicBlock
from trainer import *
from utils import *

# Load the dataset
test_loader = DataLoader(minicifar_test, batch_size=32, shuffle=True, num_workers=2)

# Load the model
model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=4)
model.load_state_dict(torch.load("lab1/resnet18.pth"))
model = to_device(model)
loss_fn = torch.nn.CrossEntropyLoss()

# Normal pretrained model
_, accuracy, class_accuracy = test_once(model, test_loader, loss_fn)
print("Normal model")
print("accuracy={}, class_accuracy={}".format(round(accuracy, 2), [round(x, 2) for x in class_accuracy]))

# Pruned pretrained model
pruned_model = copy.deepcopy(model)
for m in pruned_model.modules():
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        prune.l1_unstructured(m, name="weight", amount=0.4)
        prune.remove(m, "weight")

_, accuracy, class_accuracy = test_once(pruned_model, test_loader, loss_fn)
print("Pruned model")
print("accuracy={}, class_accuracy={}".format(round(accuracy, 2), [round(x, 2) for x in class_accuracy]))