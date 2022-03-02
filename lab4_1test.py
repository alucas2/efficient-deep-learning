import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from minicifar_test import minicifar_test
from lab1_model import ResNet, BasicBlock
from trainer import *
from utils import *
import matplotlib
matplotlib.use('Agg')
import numpy as np

# Load the dataset
test_loader = DataLoader(minicifar_test, batch_size=32, shuffle=True, num_workers=2)

# Load the model
resnet = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[16, 32, 64, 128], num_classes=4)
resnet.load_state_dict(torch.load("models/MCDAresnet18.pth"))
resnet = to_device(resnet)

# Compute accuracy
loss_fn = torch.nn.CrossEntropyLoss()
_, accuracy,_= test_once(resnet, test_loader, loss_fn)

print("accuracy={:.3f}".format(accuracy))
from matplotlib import pyplot as plt 

### Let's do a figure for each batch
f = plt.figure(figsize=(10,10))

for i,(data,target) in enumerate(test_loader):
    
    data = (data.numpy())
    print(data.shape)
    plt.subplot(2,2,1)
    plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,2)
    plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,3)
    plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,4)
    plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))

    break

f.savefig('train_DA1.png')
