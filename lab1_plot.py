import matplotlib.pyplot as plt
import numpy as np
from trainer import TrainMetrics

metrics = TrainMetrics()
metrics.load("lab1/resnet18_trainlog.csv")

plt.figure("Loss")
plt.plot(metrics["epoch"], metrics["train_loss"], label="Train")
plt.plot(metrics["epoch"], metrics["valid_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()

plt.figure("Accuracy")
plt.plot(metrics["epoch"], metrics["train_accuracy"], label="Train accuracy")
plt.plot(metrics["epoch"], metrics["valid_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()

plt.show()