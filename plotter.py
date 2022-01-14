import matplotlib.pyplot as plt
import numpy as np

overlaps = np.loadtxt("annealing_sign_problem/overlap.txt")

indices_ov = np.array([iloop for iloop in range(len(overlaps))])

fig, ax = plt.subplots()
ax.plot(indices_ov, overlaps, 'o')
ax.set(xlabel='Aeon', ylabel='Overlap')
ax.grid()
fig.savefig("overlaps.pdf")
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
