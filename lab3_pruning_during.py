import torch
import torch.nn.utils.prune as prune
from torch.utils.data.dataloader import DataLoader
from minicifar import minicifar_train, minicifar_test, train_sampler, valid_sampler
from lab1_model import ResNet, BasicBlock
from trainer import *
from utils import *
import numpy as np

# Load the datasets
train_loader = DataLoader(minicifar_train, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(minicifar_train, batch_size=32, sampler=valid_sampler)
test_loader = DataLoader(minicifar_test, batch_size=32, shuffle=True, num_workers=2)

# Create the model
model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=4)
model.load_state_dict(torch.load("lab1/resnet18.pth"))
model = to_device(model)
loss_fn = torch.nn.CrossEntropyLoss()

for fine_tune_step in range(20):
    # Prune the model
    iteration_amount = 0.2
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            prune.l1_unstructured(m, name="weight", amount=iteration_amount)
    
    amount = pruning_amount(model)
    print("Pruning iteration {}: amount={:.2f}. Fine tuning...".format(fine_tune_step, amount))

    # Train the model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=torch.nn.CrossEntropyLoss()
    )
    trainer.train(num_epochs=4)


print("Iterative pruning finished")

for m in model.modules():
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        prune.remove(m, name="weight")

_, accuracy, class_accuracy = test_once(model, test_loader, loss_fn)
amount = pruning_amount(model)

print("amount={:.2}, accuracy={:.2f}, class_accuracy={}".format(
    amount, accuracy, [round(x, 2) for x in class_accuracy]
))