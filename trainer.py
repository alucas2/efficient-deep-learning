import torch
import tqdm

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

    return train_loss_sum / len(data_loader) # average loss


def test_once(model, data_loader, loss_fn):
    """Test a model on a dataset. Returns the average loss, accuracy and accuracy for each class"""
    model.eval()
    valid_loss_sum = 0
    num_classes = model(to_device(iter(data_loader).next()[0])).size(1)
    num_correct = [0] * num_classes
    num_tested = [0] * num_classes

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
            for i in range(num_classes):
                num_tested[i] += (target == i).sum().item()
                num_correct[i] += ((target == i) & (target == predicted)).sum().item()

    return (
        valid_loss_sum / len(data_loader), # loss
        sum(num_correct) / sum(num_tested), # accuracy
        [nc / nt for nc, nt in zip(num_correct, num_tested)] # accuracy for each class
    )


class TrainMetrics:
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.valid_loss = []
        self.accuracy = []
        self.class_accuracy = []


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
            prograssbar_wrap = tqdm.tqdm(self.train_loader, desc="Epoch {}".format(epoch_id))
            train_loss = train_once(self.model, prograssbar_wrap, self.optimizer, self.loss_fn)
            
            # Validate
            valid_loss, accuracy, class_accuracy = test_once(self.model, self.valid_loader, self.loss_fn)
            
            # Save metrics
            output.epochs.append(epoch_id)
            output.accuracy.append(accuracy)
            output.train_loss.append(train_loss)
            output.valid_loss.append(valid_loss)
            output.class_accuracy.append(class_accuracy)
            print("Epoch {}: loss={}, validation_loss={}, accuracy={}, class_accuracy={}".format(
                epoch_id,
                round(train_loss, 3),
                round(valid_loss, 3),
                round(accuracy, 3),
                [round(x, 3) for x in class_accuracy]
            ))

        # Training finished
        return output
