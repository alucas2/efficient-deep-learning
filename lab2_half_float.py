import torch
from torch.utils.data.dataloader import DataLoader
from minicifar import minicifar_test
from lab1_model import ResNet, BasicBlock
from trainer import *
from utils import *

test_loader = DataLoader(minicifar_test, batch_size=32, shuffle=True, num_workers=2)

model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=4)
model.load_state_dict(torch.load("lab1/resnet18.pth"))
model = to_device(model.half())

loss_fn = torch.nn.CrossEntropyLoss()
loss, accuracy = test_once(model, test_loader, loss_fn)

print("loss={:.3f}, accuracy={:.3f}".format(loss, accuracy))