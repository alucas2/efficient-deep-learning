import torch
import torchinfo
from minicifar import minicifar_train, train_sampler, valid_sampler
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from trainer import *
import numpy as np

# Load the datasets
train_loader = DataLoader(minicifar_train, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(minicifar_train, batch_size=32, sampler=valid_sampler)

# Create the model
model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[16, 32, 64, 128], num_classes=4)
model = to_device(model)

# Count the parameters
num_parameters = sum(p.numel() for p in model.parameters())
print("Number of parameters: {}".format(num_parameters))
torchinfo.summary(model, input_size=(32, 3, 32, 32))

# Train the model
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    loss_fn=torch.nn.CrossEntropyLoss()
)
metrics = trainer.train(num_epochs=100)

# Save the model
torch.save(model.state_dict(), "lab1/thinresnet18.pth")

# Save the metrics
np.savetxt("lab1/thinresnet18_trainlog.csv",
    np.array([metrics.epochs, metrics.train_loss, metrics.train_accuracy, metrics.valid_loss, metrics.valid_accuracy]).T,
    header="epoch, train_loss, train_accuracy, valid_loss, valid_accuracy"
)

