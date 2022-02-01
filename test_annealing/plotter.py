import matplotlib.pyplot as plt
import numpy as np

sa_accuracy = np.loadtxt("SA_accuracy.txt")
overlaps = np.loadtxt("overlaps.txt")
accuracy = np.loadtxt("accuracies.txt")

indices_sa = np.array([iloop for iloop in range(len(sa_accuracy))])
indices_ov = np.array([iloop for iloop in range(len(overlaps))])
indices_ac = np.array([iloop for iloop in range(len(accuracy))])

fig, ax = plt.subplots()
ax.scatter(accuracy, overlaps)
ax.set(xlabel='Accuracy', ylabel='Overlap')
ax.grid()
fig.savefig("overlap_vs_accuracy.pdf")
plt.show()

fig, ax = plt.subplots()
ax.plot(indices_sa, sa_accuracy, 'o')
ax.set(xlabel='Aeon', ylabel='SA accuracy')
ax.grid()
fig.savefig("Sa_vs_aeon.pdf")
plt.show()

fig, ax = plt.subplots()
ax.plot(indices_ov, overlaps, 'o')
ax.set(xlabel='Aeon', ylabel='Overlap')
ax.grid()
fig.savefig("overlaps_vs_aeon.pdf")
plt.show()

fig, ax = plt.subplots()
ax.plot(indices_ac, accuracy, 'o')
ax.set(xlabel='Aeon', ylabel='Accuracy')
ax.grid()
fig.savefig("accuracy_vs_aeon.pdf")
plt.show()

"""datasize = np.loadtxt("SAacc_vs_dataset/dataset_size.txt")
mc_size = np.loadtxt("SAacc_vs_dataset/chain_length.txt")
sa_accuracy = np.loadtxt("SAacc_vs_dataset/SA_accuracy.txt")

fig, ax = plt.subplots()
ax.plot(datasize, sa_accuracy,'o')
ax.set(xlabel='Number of data points', ylabel='SA accuracy')
ax.grid()
fig.savefig("SA_vs_data.pdf")
plt.show()

fig, ax = plt.subplots()
ax.plot(mc_size, sa_accuracy,'o')
ax.set(xlabel='Markov chain length', ylabel='SA accuracy')
ax.grid()
fig.savefig("SA_vs_MC.pdf")
plt.show()

fig, ax = plt.subplots()
ax.plot(mc_size, datasize, 'o')
ax.set(xlabel='Markov chain length', ylabel='Number of data points')
ax.grid()
fig.savefig("data_vs_MC.pdf")
plt.show()"""
