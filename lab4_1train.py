import torch
import torchinfo
import matplotlib
matplotlib.use('Agg')
#from minicifar import minicifar_train, train_sampler, valid_sampler
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from trainer import *
import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

# Load the datasets
trainloader = DataLoader(c10train,batch_size=4,shuffle=False) 
testloader= DataLoader(c10test, batch_size=4,shuffle=True, num_workers=2)

# Create the model
model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[16, 32, 64, 128], num_classes=10)
model = to_device(model)

# Count the parameters
# num_parameters = sum(p.numel() for p in model.parameters())
# print("Number of parameters: {}".format(num_parameters))
# torchinfo.summary(model, input_size=(4, 3, 32, 32))

# Train the model
trainer = Trainer(
    model=model,
    train_loader=trainloader,
    valid_loader=testloader,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
    loss_fn=torch.nn.CrossEntropyLoss()
)
metrics = trainer.train(num_epochs=50)

# Save the model
torch.save(model.state_dict(), "lab4/DAresnet18.pth")

