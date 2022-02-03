import torch

def to_device(thing):
    """Host a tensor on GPU if available"""
    if torch.cuda.is_available():
        return thing.cuda()
    else:
        return thing


def train_once(model, data_loader, optimizer, loss_fn):
    """Train a model on a dataset. Returns the average loss"""
    model.train()
    train_loss_sum = 0
    for batch_id, (x, target) in enumerate(data_loader):
        x = to_device(x)
        target = to_device(target)

        # Reset the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        y = model(x)
        loss = loss_fn(y, target)
        train_loss_sum += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    return train_loss_sum / len(data_loader)


def test_once(model, data_loader, loss_fn):
    """Test a model on a dataset. Returns the average loss and accuracy"""
    model.eval()
    valid_loss_sum = 0
    num_correct = 0
    num_tested = 0
    with torch.no_grad():
        for batch_id, (x, target) in enumerate(data_loader):
            x = to_device(x)
            target = to_device(target)

            # Forward pass
            y = model(x)
            loss = loss_fn(y, target)
            valid_loss_sum += loss.item()

            # Count accuracy
            _, predicted = torch.max(y.data, 1)
            num_tested += target.size(0)
            num_correct += (predicted == target).sum().item()

    return valid_loss_sum / len(data_loader), num_correct / num_tested


class TrainMetrics:
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.valid_loss = []
        self.accuracy = []


class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, loss_fn):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, num_epochs):
        """Train a validate on a number of epochs"""
        output = TrainMetrics()
        for epoch_id in range(num_epochs):
            # Train
            train_loss = train_once(self.model, self.train_loader, self.optimizer, self.loss_fn)
            
            # Validate
            valid_loss, accuracy = test_once(self.model, self.valid_loader, self.loss_fn)
            
            # Save metrics
            output.epochs.append(epoch_id)
            output.accuracy.append(accuracy)
            output.train_loss.append(train_loss)
            output.valid_loss.append(valid_loss)
            print("Epoch {}: loss={:.3f}, validation_loss={:.3f}, accuracy={:.3f}".format(
                epoch_id, train_loss, valid_loss, accuracy
            ))

        # Training finished
        return output
