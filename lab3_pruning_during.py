import torch
import torch.nn.utils.prune as prune
from torch.utils.data.dataloader import DataLoader
from minicifar import minicifar_train, train_sampler, valid_sampler
from lab1_model import ResNet, BasicBlock
from trainer import *
from utils import *
import numpy as np

SOFT_PRUNING = True

# Load the datasets
train_loader = DataLoader(minicifar_train, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(minicifar_train, batch_size=32, sampler=valid_sampler)

# Create the model
model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[16, 32, 64, 128], num_classes=4)
model.load_state_dict(torch.load("lab1/thinresnet18.pth"))
model = to_device(model)

amounts = []
accuracies = []
for fine_tune_step in range(50):
    
    if SOFT_PRUNING:
        # Apply the pruning in case of soft pruning
        iteration_amount = 0.9
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                prune.l1_unstructured(m, name="weight", amount=iteration_amount)
                prune.remove(m, "weight")
    else:
        # Prune the model
        iteration_amount = 0.1
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                prune.l1_unstructured(m, name="weight", amount=iteration_amount)

    amount = pruning_amount(model)
    amounts.append(amount)
    print("Pruning iteration {}: amount={:.2f}. Fine tuning...".format(fine_tune_step, amount))

    # Train the model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=torch.nn.CrossEntropyLoss()
    )
    trainer.train(num_epochs=2)

    # Test
    _, accuracy, _ = test_once(model, valid_loader, trainer.loss_fn)
    accuracies.append(accuracy)


print("Iterative pruning finished")

if not SOFT_PRUNING:
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            prune.remove(m, name="weight")

np.savetxt("lab3/pruning_during.csv", np.array([amounts, accuracies]).T, header="amount, accuracy")
