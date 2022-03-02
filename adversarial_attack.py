import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from data import *
from trainer2 import *
import numpy as np
import matplotlib.pyplot as plt

num_classes = 4
model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[64, 128, 256, 512], num_classes=num_classes)
model.load_state_dict(torch.load("models/normalresnet18_for_minicifar_mixup.pth"))
model = model.cuda()

test_dataset = get_minicifar_test(TRANSFORM_TEST)
loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

x, _ = next(iter(loader))
x = x.cuda()
x.requires_grad = True
desired_class = 3 # cat
target = torch.Tensor([[i == desired_class for i in range(num_classes)]]).cuda()
optimizer = torch.optim.Adam([x])
loss_fn = CrossEntropyLoss()

# Show the original image
image = TRANSFORM_TEST_INV(x[0].cpu())
image.save("data/adversial_attack_original.png")
plt.figure()
plt.imshow(image)

# Transform the image
for i in range(500):
    optimizer.zero_grad()
    y = model(x)
    loss = loss_fn(y, target)
    loss.backward()
    [y_class] = torch.max(y.data, dim=1).indices
    if i % 100 == 0:
        print(f"iteration={i}, loss={loss.item():.3f}, prediction={y_class}")
    optimizer.step()

# Show the transformed image
image = TRANSFORM_TEST_INV(x[0].cpu())
image.save("data/adversial_attack_noisy.png")
plt.figure()
plt.imshow(image)
plt.show()
