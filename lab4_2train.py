import torch
import matplotlib.pyplot as plt
from minicifar import minicifar_train, train_sampler, valid_sampler,minicifar_test
from torch.utils.data.dataloader import DataLoader
from lab1_model import *
from trainer import *
from binaryconnect import *

num_epochs= 50
batch_size=32

# Load the datasets
train_loader = DataLoader(minicifar_train, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(minicifar_train, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(minicifar_test, batch_size=batch_size, shuffle=True, num_workers=2)

model = ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[16, 32, 64, 128], num_classes=4)
model= to_device(model)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
n_total_steps=len(train_loader)

for epoch in range(num_epochs):
    # Train
    
    model.train()
    model.train()
    train_loss_sum = 0
    num_correct = 0
    num_tested = 0
    L=len(train_loader)
    i=0

    for batch_id, (x, target) in enumerate(train_loader):
        
        (Xi,targeti)= train_loader[L-x[i]]
        i=i+1
        lamb= random.random()
        x = to_device(x)
        target = to_device(target)
        Xi = to_device(Xi)
        targeti = to_device(targeti)
        lamb= to_device(lamb)
        Mixed_up_data= x + (1-lamb)*Xi

        # Reset the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        y = model(Mixed_up_data)
        Mixed_loss= lamb* loss_fn(y, target)+ (1-lamb)*loss_fn(y, targeti)

        # Backward pass
        Mixed_loss.backward()
        optimizer.step()


        if (i+1) % 100==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}],Loss: {loss.item():.4f}')

    # Test


    model.eval()

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct=[0 for i in range(10)]
        n_class_samples=[0 for i in range(10)]
        for x, labels in valid_loader:
            x = to_device(x)
            labels = to_device(labels)

            # Forward pass
            
          
            y=model(x)
            loss = loss_fn(y, labels)

            # Count accuracy
            _, predicted = torch.max(y.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label=labels[i]
                pred=predicted[i]
                if (label==pred):
                    n_class_correct[label]+=1
                n_class_samples[label]+=1
        acc=100.0* n_correct/n_samples
        print(f'Accuracy of network: {acc}%')

for i in range(4):
    acc=100.0* n_class_correct[i]/n_class_samples[i]
    print(f'Accuracy of classe {[i]}: {acc}%')
