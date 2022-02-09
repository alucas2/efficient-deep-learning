import torch
import matplotlib.pyplot as plt
from minicifar import minicifar_train, train_sampler, valid_sampler
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from trainer import *

# Load the datasets
train_loader = DataLoader(minicifar_train, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(minicifar_train, batch_size=32, sampler=valid_sampler)


fig_loss = plt.figure(1)
fig_accuracy = plt.figure(2)

for model_name, model in [
    ("ResNet18", ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=4))
]:
    # Create the model
    model = to_device(model)

    # Count the parameters
    num_parameters = sum(p.numel() for p in model.parameters())
    print("Number of parameters ({}): {}".format(model_name, num_parameters))

    # Train the model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=torch.nn.CrossEntropyLoss()
    )
    metrics = trainer.train(num_epochs=50)

    fig_loss.gca().plot(metrics.epochs, metrics.train_loss, label="Train loss ({})".format(model_name))
    fig_loss.gca().plot(metrics.epochs, metrics.valid_loss, label="Validation loss ({})".format(model_name))
    fig_accuracy.gca().plot(metrics.epochs, metrics.accuracy, label="Accuracy ({})".format(model_name))

# Save the model
torch.save(model.state_dict(), "lab1/resnet18.pth")

# Plot the metrics
fig_loss.legend()
fig_loss.savefig("lab1/resnet_train_loss.png")

fig_accuracy.legend()
fig_accuracy.savefig("lab1/resnet_train_accuracy")

