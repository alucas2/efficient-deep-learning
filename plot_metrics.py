import matplotlib.pyplot as plt
import sys
import numpy as np
from trainer2 import TrainMetrics

filename = "logs/thinresnet18_for_minicifar.csv"
if len(sys.argv) == 2:
    filename = sys.argv[1]

metrics = TrainMetrics()
metrics.load(filename)

plt.figure(f"Loss - {filename}")
plt.plot(metrics["epoch"], metrics["train_loss"], label="Train")
plt.plot(metrics["epoch"], metrics["valid_loss"], label="Validation")
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()

best_accuracy = np.max(metrics["valid_accuracy"])
plt.figure(f"Accuracy - {filename}")
plt.plot(metrics["epoch"], metrics["train_accuracy"], label="Train")
plt.plot(metrics["epoch"], metrics["valid_accuracy"], label="Validation")
plt.title(f"Accuracy vs Epoch (best={best_accuracy})")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()

plt.show()