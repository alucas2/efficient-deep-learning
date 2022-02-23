import matplotlib.pyplot as plt
import numpy as np

[epochs, train_loss, train_accuracy, valid_loss, valid_accuracy] = np.loadtxt("lab1/thinresnet18_trainlog.csv").T

# Plot
plt.figure(1)
plt.plot(epochs, train_loss, label="Train")
plt.plot(epochs, valid_loss, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.figure(2)
plt.plot(epochs, train_accuracy, label="Train accuracy")
plt.plot(epochs, valid_accuracy, label="Validation accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Loss")
plt.legend()

plt.show()