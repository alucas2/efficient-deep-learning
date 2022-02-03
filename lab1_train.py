import torch
import matplotlib.pyplot as plt
from minicifar import minicifar_train, train_sampler, valid_sampler
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from trainer import *

# Load the datasets
train_loader = DataLoader(minicifar_train, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(minicifar_train, batch_size=32, sampler=valid_sampler)

# Create the model
resnet = ResNet(BasicBlock, [2, 2, 2, 2])
resnet = to_device(resnet)

# Train the model
trainer = Trainer(
    model=resnet,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=torch.optim.SGD(resnet.parameters(), lr=0.01),
    loss_fn=torch.nn.CrossEntropyLoss()
)
metrics = trainer.train(num_epochs=15)

# Save the model
torch.save(resnet.state_dict(), "lab1_resnet.pth")

# Plot the metrics
plt.plot(metrics.epochs, metrics.train_loss, label="Train loss")
plt.plot(metrics.epochs, metrics.valid_loss, label="Validation loss")
plt.legend()
plt.savefig("lab1_resnet_train.png")
