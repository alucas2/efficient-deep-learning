import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from minicifar import minicifar_test
from lab1_model import ResNet, BasicBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet = ResNet(BasicBlock, [2, 2, 2, 2])
resnet.to(device)
resnet.eval()

PATH = './lab1_resnet.pth'
resnet.load_state_dict(torch.load(PATH))

testloader = torch.utils.data.DataLoader(minicifar_test, batch_size=32, shuffle=False, num_workers=2)

dataiter = iter(testloader)
images, labels = dataiter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

correct = 0
total = 0
with torch.no_grad():  # torch.no_grad for TESTING
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# plt.show()