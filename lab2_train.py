import torch
import matplotlib.pyplot as plt
from minicifar import minicifar_train, train_sampler, valid_sampler
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from trainer import *
from binaryconnect import BC

# Load the datasets
train_loader = DataLoader(minicifar_train, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(minicifar_train, batch_size=32, sampler=valid_sampler)


fig_loss = plt.figure(1)
fig_accuracy = plt.figure(2)

for model_name, model in [
    ("Binarized_ResNet18", ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=4))
]:
    # Create the model
    model = to_device(model)
    binarizer = BC(model)

    def forward_pre_hook(_model, _input):
        binarizer.clip()
        binarizer.binarization()
    def backward_hook(_model, _grad_input, _grad_output):
        binarizer.restore()

    model.register_forward_pre_hook(forward_pre_hook)
    model.register_backward_hook(backward_hook)

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
    metrics = trainer.train(num_epochs=100)

    fig_loss.gca().plot(metrics.epochs, metrics.train_loss, label="Train loss ({})".format(model_name))
    fig_loss.gca().plot(metrics.epochs, metrics.valid_loss, label="Validation loss ({})".format(model_name))
    fig_accuracy.gca().plot(metrics.epochs, metrics.accuracy, label="Accuracy ({})".format(model_name))

# Save the model
torch.save(model.state_dict(), "lab2/binarized_resnet18.pth")

# Plot the metrics
fig_loss.legend()
fig_loss.savefig("lab2/binarized_resnet_train_loss.png")

fig_accuracy.legend()
fig_accuracy.savefig("lab2/binarized_resnet_train_accuracy")

