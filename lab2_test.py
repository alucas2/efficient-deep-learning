import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from minicifar import minicifar_test
from lab1_model import *
from utils import *
from trainer import *
from binaryconnect import BC



# Load the dataset
test_loader = DataLoader(minicifar_test, batch_size=32, shuffle=True, num_workers=2)

# Load the model


model= ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=4)
model.load_state_dict(torch.load("lab2/binarized_resnet18.pth"))

model= to_device(model)

loss_fn = torch.nn.CrossEntropyLoss()

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct=[0 for i in range(10)]
    n_class_samples=[0 for i in range(10)]
    for x, labels in test_loader:
        x = to_device(x)
        labels = to_device(labels)

        # Forward pass
            
        y = model(x)
        
        loss = loss_fn(y, labels)

        # Count accuracy
        _, predicted = torch.max(y.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(32):
            label=labels[i]
            pred=predicted[i]
            if (label==pred):
                n_class_correct[label]+=1
            n_class_samples[label]+=1
    acc=100.0* n_correct/n_samples
    print(f'Accuracy of network: {acc}%')

for i in range(4):
    acc=100.0* n_class_correct[i]/n_class_samples[i]
    print(f'Accuracy of classe {[i]}: {acc}%')


# # Compute accuracy
# loss_fn = torch.nn.CrossEntropyLoss()
# _, accuracy = test_once(model, test_loader, loss_fn)

