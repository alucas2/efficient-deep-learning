import torch
import matplotlib.pyplot as plt
from minicifar import minicifar_train, train_sampler, valid_sampler
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from trainer import *
from binaryconnect import *

# Load the datasets
train_loader = DataLoader(minicifar_train, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(minicifar_train, batch_size=32, sampler=valid_sampler)

model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=4)
model = to_device(model)
model = BC(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch_id in range(5):
    # Train
    model.train()
    train_loss_sum = 0
    for (x, target) in tqdm.tqdm(train_loader, desc="Epoch {}".format(epoch_id)):
        x = to_device(x)
        target = to_device(target)

        # Reset the parameter gradients
        optimizer.zero_grad()

        # Binarize and forward pass
        model.binarization()
        y = model(x)
        loss = loss_fn(y, target)
        train_loss_sum += loss.item()

        # Backward pass and restore
        loss.backward()
        model.restore()
        optimizer.step()
        model.clip()

    train_loss = train_loss_sum / len(train_loader)

    # Test
    model.binarization()
    valid_loss, accuracy = test_once(model, valid_loader, loss_fn)
    print("Epoch {}: loss={:.3f}, validation_loss={:.3f}, accuracy={:.3f}".format(
        epoch_id, train_loss, valid_loss, accuracy
    ))
