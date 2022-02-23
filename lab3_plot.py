import matplotlib.pyplot as plt
import numpy as np

[amount, accuracy] = np.loadtxt("lab3/pruning_during.csv").T

# plt.plot(accuracy)
# plt.xlabel("Iteration")
# plt.ylabel("Accuracy")

plt.plot(amount, accuracy)
plt.xlabel("Pruning ratio")
plt.ylabel("Accuracy")
plt.grid()

plt.show()