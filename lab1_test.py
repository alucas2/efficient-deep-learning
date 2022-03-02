import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from lab1_model import ResNet, BasicBlock
from data import *
from trainer2 import *
from utils import *

# Load the dataset
test_loader = DataLoader(get_minicifar_test(TRANSFORM_TEST), batch_size=32, shuffle=True, num_workers=2)

# Load the model
resnet = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=4)
resnet.load_state_dict(torch.load("lab1/resnet18.pth"))
batch_preprocess = []

if torch.cuda.is_available():
    resnet = resnet.cuda()
    batch_preprocess.append(batch_to_cuda)

# Compute accuracy
loss_fn = CrossEntropyLoss()
_, accuracy, _ = test_once(resnet, test_loader, loss_fn, batch_preprocess)

print("accuracy={:.3f}".format(accuracy))

# Show some examples
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
dataiter = iter(test_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
plt.savefig("lab1/resnet_examples.png")
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(32)))