import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from minicifar import minicifar_test
from lab1_model import ResNet, BasicBlock
from trainer import *
from utils import *

# Load the dataset
test_loader = DataLoader(minicifar_test, batch_size=32, shuffle=True, num_workers=2)

# Load the model
resnet = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=4)
resnet.load_state_dict(torch.load("lab1/resnet18.pth"))
resnet = to_device(resnet)

# Compute accuracy
loss_fn = torch.nn.CrossEntropyLoss()
_, accuracy = test_once(resnet, test_loader, loss_fn)

print("accuracy={:.3f}".format(accuracy))

# Show some examples
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
dataiter = iter(test_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
plt.savefig("lab1/resnet_examples.png")
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(32)))