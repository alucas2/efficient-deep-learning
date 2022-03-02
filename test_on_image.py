import torch
import torch.nn.functional as F
import sys
from data import *
from lab1_model import *
import matplotlib.pyplot as plt
from PIL import Image

model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[64, 128, 256, 512], num_classes=4)
model.load_state_dict(torch.load("models/normalresnet18_for_minicifar_mixup.pth"))

for i in range(1, len(sys.argv)):
    image_path = sys.argv[i]
    image = Image.open(image_path).convert("RGB")

    x = TRANSFORM_TEST(image)[None]
    [y] = model(x)
    _, y_class = torch.max(y, dim=0)

    plt.subplot(1, len(sys.argv) - 1, i)
    plt.imshow(image)
    plt.title(f"{CLASS_NAMES[y_class]}")

plt.show()
