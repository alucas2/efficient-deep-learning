import torch
import torchinfo
import torch.nn.utils.prune as prune
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from data import *
from trainer2 import *
from utils import *
import numpy as np
import copy

BASE_MODEL_PATH = "models/normalresnet18_for_cifar10_mixup.pth"
USE_CIFAR10 = True
USE_THIN_RESNET18 = False
USE_RESNET20 = True
USE_HALF = False
USE_MIXUP = False
USE_AUTOAUGMENTED= False 
BATCH_SIZE = 32

dataset_description = []

if USE_AUTOAUGMENTED:
    transform_train = TRANSFORM_TRAIN_AUTOAUGMENTED
    transform_test = TRANSFORM_TEST
    dataset_description.append("autoaugmented")
else:
    transform_train = TRANSFORM_TRAIN
    transform_test = TRANSFORM_TEST

# Load the datasets
if USE_CIFAR10:
    train_dataset = get_cifar10_train(transform_train)
    test_dataset = get_cifar10_test(transform_test)
    num_classes = 10
    dataset_description.append("cifar10")
else:
    train_dataset = get_minicifar_train(transform_train)
    test_dataset = get_minicifar_test(transform_test)
    num_classes = 4
    dataset_description.append("minicifar")

# Create the model
if USE_THIN_RESNET18:
    model_name = "thinresnet18"
    model = make_thinresnet18(num_classes)
elif USE_RESNET20:
    model_name = "resnet20"
    model = make_resnet20(num_classes)
else:
    model_name = "normalresnet18"
    model = make_resnet18(num_classes)

# Load the base model
model.load_state_dict(torch.load(BASE_MODEL_PATH))
dataset_description.append("pruned")

# --------------------------------------------------------------------------------------------------------

train_preprocess = []
test_preprocess = []

if torch.cuda.is_available():
    model = model.cuda()
    train_preprocess.append(batch_to_cuda)
    test_preprocess.append(batch_to_cuda)

if USE_HALF:
    dataset_description.append("half")
    model = model.half()
    train_preprocess.append(batch_to_half)
    test_preprocess.append(batch_to_half)

if USE_MIXUP: # the train accuracy is garbage when mixup is activated
    dataset_description.append("mixup")
    train_preprocess.append(batch_mixup)

# --------------------------------------------------------------------------------------------------------

file_name = model_name + "_for_" + '_'.join(dataset_description)
print(file_name)

# Load data
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training settings
loss = CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
lr_scheduler = None
trainer = Trainer(model, train_loader, valid_loader, loss, optimizer, lr_scheduler, train_preprocess, test_preprocess)

# Train the model
best_pruned_model = None
for fine_tune_step in range(10):
    # Prune the model
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            prune.l1_unstructured(m, name="weight", amount=0.3)

    metrics, _ = trainer.train(num_epochs=2)
    if best_pruned_model is None or metrics["valid_accuracy"][-1] > 0.9:
        best_pruned_model = copy.deepcopy(model)

# Save the model and metrics
metrics.save(f"logs/{file_name}.csv")
torch.save(best_pruned_model.state_dict(), f"models/{file_name}.pth")

# Save the model summary
# with open(f"models/{model_name}.txt", "w") as f:
#     f.write(str(torchinfo.summary(
#         model, input_size=(BATCH_SIZE, 3, 32, 32), dtypes=[torch.half] if USE_HALF else [torch.float]
#     )))