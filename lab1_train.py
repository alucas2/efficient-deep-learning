from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader

trainloader = DataLoader(minicifar_train,batch_size=4,sampler=train_sampler)
validloader = DataLoader(minicifar_train,batch_size=4,sampler=valid_sampler)

import torch
import torch.optim as optim
import torch.nn as nn
from lab1_model import ResNet, BasicBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet = ResNet(BasicBlock, [2, 2, 2, 2])
resnet.to(device)

optimizer = optim.SGD(resnet.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
n_epochs = 15

for epoch in range(n_epochs):  # loop over the dataset multiple times

    # resnet.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
            
    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
    running_loss = 0.0

PATH = './lab1_resnet.pth'
torch.save(resnet.state_dict(), PATH)