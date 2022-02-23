import torch
import matplotlib.pyplot as plt
from minicifar import minicifar_train, train_sampler, valid_sampler,minicifar_test
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from trainer import *
from binaryconnect import *

num_epochs= 50
batch_size=32
learning_rate=0.01

# Load the datasets
train_loader = DataLoader(minicifar_train, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(minicifar_train, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(minicifar_test, batch_size=batch_size, shuffle=True, num_workers=2)

model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=4)
modelbc = BC(model)
modelbc.model = to_device(modelbc.model)
optimizer = torch.optim.SGD(modelbc.model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
n_total_steps=len(train_loader)

for epoch in range(num_epochs):
    # Train
    modelbc.model.train()


    for i, (x, target) in enumerate(train_loader): 
        
        xn= x.clone ()
        xnc= x.clone ()

        for i in range(len(x)):
            xn[i]=2.0*(x[i].data >= 0)-1.0
            xnc[i]=nn.functional.hardtanh(xn[i].data, min_val=-1.0, max_val=1.0)

        xnc = to_device(xnc)
        target = to_device(target)

        # Reset the parameter gradients
        optimizer.zero_grad()

        # Binarize and forward pass
        modelbc.binarization()
        y = modelbc.model(xnc)
        loss = loss_fn(y, target)

        # Backward pass and restore
        
        loss.backward()
        modelbc.restore()
        optimizer.step()
        modelbc.clip()

        if (i+1) % 100==0:
            print(f'Epoch [{epoch+1}/{num_epochs}],Loss: {loss.item():.4f}')


    # Test

    # modelbc.model.eval()
    # #model.eval()
    # modelbc.binarization()
    # with torch.no_grad():
    #     n_correct = 0
    #     n_samples = 0
    #     n_class_correct=[0 for i in range(10)]
    #     n_class_samples=[0 for i in range(10)]
    #     for x, labels in valid_loader:
    #         x = to_device(x)
    #         labels = to_device(labels)

    #         # Forward pass
            
    #         y = modelbc.model(x)
    #         #y=model(x)
    #         loss = loss_fn(y, labels)

    #         # Count accuracy
    #         _, predicted = torch.max(y.data, 1)
    #         n_samples += labels.size(0)
    #         n_correct += (predicted == labels).sum().item()

    #         for i in range(batch_size):
    #             label=labels[i]
    #             pred=predicted[i]
    #             if (label==pred):
    #                 n_class_correct[label]+=1
    #             n_class_samples[label]+=1
    #     acc=100.0* n_correct/n_samples
    #     print(f'Accuracy of network: {acc}%')

# for i in range(4):
#     acc=100.0* n_class_correct[i]/n_class_samples[i]
#     print(f'Accuracy of classe {[i]}: {acc}%')



    # modelbc.binarization()
    # valid_loss, accuracy = test_once(modelbc.model, valid_loader, loss_fn)
    # print("Epoch {}: loss={:.3f}, validation_loss={:.3f}, accuracy={:.3f}".format(
    #     epoch_id, train_loss, valid_loss, accuracy
    # ))