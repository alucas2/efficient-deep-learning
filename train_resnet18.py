import torch
import torchinfo
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from data import *
from trainer2 import *
import numpy as np

USE_CIFAR10 = True
USE_THIN_RESNET18 = False
USE_HALF = False
USE_MIXUP = True
BATCH_SIZE = 32

# Load the datasets
if USE_CIFAR10:
    train_dataset = get_cifar10_train(TRANSFORM_TRAIN)
    test_dataset = get_cifar10_test(TRANSFORM_TEST)
    num_classes = 10
    dataset_name = "cifar10"
else:
    train_dataset = get_minicifar_train(TRANSFORM_TRAIN)
    test_dataset = get_minicifar_test(TRANSFORM_TEST)
    num_classes = 4
    dataset_name = "minicifar"

# Create the model
if USE_THIN_RESNET18:
    model_name = f"thinresnet18_for_{dataset_name}"
    model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[32, 64, 128, 256], num_classes=num_classes)
else:
    model_name = f"normalresnet18_for_{dataset_name}"
    model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[64, 128, 256, 512], num_classes=num_classes)

# --------------------------------------------------------------------------------------------------------

train_preprocess = []
test_preprocess = []

if torch.cuda.is_available():
    model = model.cuda()
    train_preprocess.append(batch_to_cuda)
    test_preprocess.append(batch_to_cuda)

if USE_HALF:
    model_name += "_half"
    model = model.half()
    train_preprocess.append(batch_to_half)
    test_preprocess.append(batch_to_half)

if USE_MIXUP: # the train accuracy is garbage when mixup is activated
    model_name += "_mixup"
    train_preprocess.append(batch_mixup)

# --------------------------------------------------------------------------------------------------------

# Save the model summary
with open(f"models/{model_name}.txt", "w") as f:
    f.write(str(torchinfo.summary(
        model, input_size=(BATCH_SIZE, 3, 32, 32), dtypes=[torch.half] if USE_HALF else [torch.float]
    )))

# Load data
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training settings
loss = CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

# Train the model
trainer = Trainer(model, train_loader, valid_loader, loss, optimizer, lr_scheduler, train_preprocess, test_preprocess)
metrics, best_model = trainer.train(num_epochs=150)

# Save the model and metrics
metrics.save(f"logs/{model_name}.csv")
torch.save(best_model.state_dict(), f"models/{model_name}.pth")
