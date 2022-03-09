from matplotlib.transforms import Transform
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import copy
from utils import *

# ----------------------------------------- Helper functions -----------------------------------------

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        log_softmax = F.log_softmax(input, dim=1)
        return -torch.sum(log_softmax * target) / input.size(0)


def train_once(model, data_loader, optimizer, loss_fn, batch_preprocess):
    """Train a model on a dataset. Returns the average loss"""
    model.train()
    train_loss_sum = 0
    num_correct = 0
    num_tested = 0

    for batch_id, (x, target) in enumerate(data_loader):
        for f in batch_preprocess:
            x, target = f(x, target)

        # Reset the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        y = model(x)
        loss = loss_fn(y, target)

        # Metrics
        train_loss_sum += loss.item()
        y_class = torch.max(y.data, dim=1).indices # get the predicted class number
        target_class = torch.max(target.data, dim=1).indices # get the actual class number
        num_tested += target_class.size(0)
        num_correct += (target_class == y_class).sum().item()
        
        # Backward pass
        loss.backward()
        optimizer.step()

    return (
        train_loss_sum / len(data_loader), # loss
        num_correct / num_tested # accuracy
    )


def test_once(model, data_loader, loss_fn, batch_preprocess):
    """Test a model on a dataset. Returns the average loss, accuracy and accuracy for each class"""
    model.eval()
    valid_loss_sum = 0
    num_classes = None
    num_correct = None
    num_tested = None

    with torch.no_grad():
        for batch_id, (x, target) in enumerate(data_loader):
            for f in batch_preprocess:
                x, target = f(x, target)

            if num_classes is None:
                num_classes = target.size(1)
                num_correct = [0] * num_classes
                num_tested = [0] * num_classes

            # Forward pass
            y = model(x)
            loss = loss_fn(y, target)
            valid_loss_sum += loss.item()

            # Count accuracy
            y_class = torch.max(y.data, dim=1).indices # get the predicted class number
            target_class = torch.max(target.data, dim=1).indices # get the actual class number
            for i in range(num_classes):
                num_tested[i] += (target_class == i).sum().item()
                num_correct[i] += ((target_class == i) & (target_class == y_class)).sum().item()

    return (
        valid_loss_sum / len(data_loader), # loss
        sum(num_correct) / sum(num_tested), # accuracy
        [nc / nt for nc, nt in zip(num_correct, num_tested)] # accuracy for each class
    )


# ----------------------------------------- Metrics -----------------------------------------

class TrainMetrics:
    def __init__(self):
        self.dict = {}

    def append_metrics(self, dict):
        for k, v in dict.items():
            if k in self.dict:
                self.dict[k].append(v)
            else:
                self.dict[k] = [v]
    
    def __getitem__(self, k):
        return self.dict[k]

    def __contains__(self, thing):
        return thing in self.dict

    def save(self, filename):
        np.savetxt(filename,
            np.array([v for v in self.dict.values()]).T,
            header=" ".join(k for k in self.dict.keys())
        )

    def load(self, filename):
        x = np.genfromtxt(filename, names=True)
        self.dict = {k: x[k] for k in x.dtype.names}


# ----------------------------------------- Trainer -----------------------------------------

class Trainer:
    def __init__(
        self, model, train_loader, valid_loader, loss, optimizer, lr_scheduler=None,
        train_preprocess=None, test_preprocess=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.loss_fn = loss
        self.lr_scheduler = lr_scheduler
        self.best_model = None
        self.best_accuracy = None
        self.train_preprocess = train_preprocess if train_preprocess else []
        self.test_preprocess = test_preprocess if test_preprocess else []
        self.metrics = TrainMetrics()
        self.current_epoch = 0

    def train(self, num_epochs) -> TrainMetrics:
        """Train a validate on a number of epochs"""
        for self.current_epoch in range(self.current_epoch+1, self.current_epoch+num_epochs):
            # Train
            prograssbar_wrap = tqdm.tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", leave=False)
            train_loss, train_accuracy = train_once(
                self.model, prograssbar_wrap, self.optimizer, self.loss_fn, self.train_preprocess
            )
            
            # Validate
            valid_loss, valid_accuracy, class_accuracy = test_once(
                self.model, self.valid_loader, self.loss_fn, self.test_preprocess
            )
            
            # Step the scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Save metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            pruning = pruning_amount(self.model)
            self.metrics.append_metrics({
                "epoch": self.current_epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_accuracy,
                "learning_rate": current_lr,
                "pruning": pruning
            })
            self.metrics.append_metrics({
                f"class_accuracy_{i}": class_accuracy[i] for i in range(len(class_accuracy))
            })

            # Save the best model
            better_model = False
            if self.best_accuracy is None or valid_accuracy > self.best_accuracy:
                better_model = True
                self.best_accuracy = valid_accuracy
                self.best_model = copy.deepcopy(self.model)

            # Show step
            print(f"epoch={self.current_epoch}, "
                + f"train_loss={train_loss:.3f}, "
                + f"train_acc={train_accuracy:.3}, "
                + f"val_loss={valid_loss:.3f}, "
                + f"val_acc={valid_accuracy:.3f}, "
                + f"class_acc={[round(x, 3) for x in class_accuracy]}, "
                + f"lr={current_lr:.3f}, "
                + f"pruning={pruning:.3f}, "
                + f"best={better_model}"
            )

        # Training finished
        return self.metrics, self.best_model
