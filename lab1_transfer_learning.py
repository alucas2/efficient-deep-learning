import torch
import matplotlib.pyplot as plt
import models_cifar100
from lab1.minicifar import minicifar_train, train_sampler, valid_sampler
from torch.utils.data.dataloader import DataLoader
from trainer import *

train_loader = DataLoader(minicifar_train, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(minicifar_train, batch_size=32, sampler=valid_sampler)

model = models_cifar100.ResNet18()
state_dict = torch.load("models_cifar100/ResNet18_model_cifar100_lr_0.01.pth")
model.load_state_dict(state_dict["net"])

for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.linear.in_features
model.linear = torch.nn.Linear(num_ftrs, 4)
model = to_device(model)

num_parameters = sum(p.numel() for p in model.parameters())
print("Number of parameters: {}".format(num_parameters))

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    loss_fn=torch.nn.CrossEntropyLoss()
)
metrics = trainer.train(num_epochs=50)

# Save the model
torch.save(model.state_dict(), "lab1_transferresnet.pth")

# Plot the metrics
plt.plot(metrics.epochs, metrics.train_loss, label="Train loss")
plt.plot(metrics.epochs, metrics.valid_loss, label="Validation loss")
plt.legend()
plt.savefig("lab1_transferresnet_train.png")

