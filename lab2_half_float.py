import torch
from torch.utils.data.dataloader import DataLoader
from minicifar import minicifar_test
from lab1_model import ResNet, BasicBlock
from trainer import *
from utils import *

test_loader = DataLoader(minicifar_test, batch_size=32, shuffle=True, num_workers=2)


model = ResNet(BasicBlock, num_blocks=[1, 1, 1, 1], num_classes=4)
model.load_state_dict(torch.load("lab1_miniresnet.pth"))
# model.half()
model = to_device(model)