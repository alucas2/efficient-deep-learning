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
# model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=4)
model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=4)
model = to_device(model)

# Count the parameters
num_parameters = sum(p.numel() for p in model.parameters())
print("Number of parameters: {}".format(num_parameters))

# Train the model
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    loss_fn=torch.nn.CrossEntropyLoss()
)
metrics = trainer.train(num_epochs=50)

# Save the model
torch.save(model.state_dict(), "lab1_miniresnet.pth")

# Plot the metrics
plt.plot(metrics.epochs, metrics.train_loss, label="Train loss")
plt.plot(metrics.epochs, metrics.valid_loss, label="Validation loss")
plt.legend()
plt.savefig("lab1_miniresnet_train.png")
