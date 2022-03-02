import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from cutout import *
from autoaugmented import *

# ----------------------------------------- Transforms -----------------------------------------

# Data augmentation is needed in order to train from scratch
TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORM_TRAIN_AUTOAUGMENTED = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
    transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
	transforms.ToTensor(), 
    Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORM_NONE = transforms.ToTensor()

CLASS_NAMES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def one_hot(num_classes):
    return lambda c: torch.Tensor([i == c for i in range(num_classes)])

def to_same_device(thing, dest_device):
    if dest_device.get_device() != -1:
        return thing.to(dest_device.get_device())
    else:
        return thing

# ----------------------------------------- Datasets -----------------------------------------

def get_subset(dataset, num_classes, subset_size):
    # collect the samples that relate to the desired class
    indices = np.array([], dtype=int)
    for i in range(num_classes):
        class_indices, = np.where(np.array(dataset.targets) == i)
        indices = np.concatenate((indices, class_indices[:subset_size//num_classes]))
    return torch.utils.data.Subset(dataset, indices)

def get_cifar10_train(transform, target_transform=one_hot(10)):
    print("Loading cifar10_train...")
    return torchvision.datasets.CIFAR10(
        "data/cifar10", train=True, download=True, transform=transform, target_transform=target_transform
    )

def get_cifar10_test(transform, target_transform=one_hot(10)):
    print("Loading cifar10_test...")
    return torchvision.datasets.CIFAR10(
        "data/cifar10", train=False, download=True, transform=transform, target_transform=target_transform
    )

def get_minicifar_train(transform):
    return get_subset(get_cifar10_train(transform, one_hot(4)), 4, 6400)

def get_minicifar_test(transform):
    return get_subset(get_cifar10_test(transform, one_hot(4)), 4, 1600)

# ----------------------------------------- Weird batch manipulations -----------------------------------------

def batch_to_cuda(images, targets):
    return images.cuda(), targets.cuda()

def batch_to_half(images, targets):
    return images.half(), targets.half()

def batch_mixup(images, targets):
    batch_size = images.size(0)
    permutation = torch.randperm(batch_size)
    shuffled_images = images[permutation]
    shuffled_targets = targets[permutation]
    mix = to_same_device(torch.Tensor([np.random.beta(0.5, 0.5) for _ in range(batch_size)]), images)
    OP1 = "i, ijkl -> ijkl"
    OP2 = "i, ij -> ij"
    mixed_images = torch.einsum(OP1, mix, images) + torch.einsum(OP1, 1-mix, shuffled_images)
    mixed_targets = torch.einsum(OP2, mix, targets) + torch.einsum(OP2, 1-mix, shuffled_targets)
    return mixed_images, mixed_targets

# ----------------------------------------- Test -----------------------------------------

def summarize_dataset(dataset, preprocess):
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt

    num_plots = (4, 10)
    loader = DataLoader(dataset, batch_size=num_plots[0] * num_plots[1], shuffle=True)

    # Print the number of images
    print(f"Size: {len(dataset)}")

    # Plot some images from the dataset
    images, targets = iter(loader).next()
    for f in preprocess:
        images, targets = f(images, targets)

    # Plot the images of the batch
    for i in range(images.size(0)):
        # Get the first and second class if applicable
        target1_percent, target1_class = torch.max(targets[i], dim=0)
        target2_percent, target2_class = torch.max(
            torch.Tensor([targets[i, j] * (j != target1_class) for j in range(targets[i].size(0))]), dim=0
        )

        image = np.array(images[i]).swapaxes(0, 1).swapaxes(1, 2)
        plt.subplot(num_plots[0], num_plots[1], i+1)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        title = f"{int(target1_percent*100)}% {CLASS_NAMES[target1_class]}"
        if target2_percent != 0:
            title += f"\n{int(target2_percent*100)}% {CLASS_NAMES[target2_class]}"
        plt.title(title)
        plt.imshow(image)
    
    plt.show()

if __name__ == "__main__":
    summarize_dataset(get_cifar10_test(TRANSFORM_TRAIN_AUTOAUGMENTED), [batch_mixup])

    

    