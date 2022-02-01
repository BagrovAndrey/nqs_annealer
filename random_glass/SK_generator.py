import lattice_symmetries as ls
import numpy as np
import h5py

np.random.seed(167453)

N = 12
mu = 0.0
sigma = 1.0

link_number = int(N*(N-1)/2)

basis = ls.SpinBasis(ls.Group([]), N, hamming_weight=N // 2)
matrix = np.array([[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]])
terms = []
for i in range(N - 1):
    for j in range(i + 1, N):

        J = np.random.normal(mu, sigma)
        print(J)

        terms.append(ls.Interaction(J * matrix, [(i, j)]))
operator = ls.Operator(basis, terms)

ground_state = ls.diagonalize(operator, 1, np.complex128)

print(ground_state[0])

"""import numpy as np

N = 16
spin_inversion = 1

link_number = int(N*(N-1)/2)

# Generate random couplings

mu = 0.0
sigma = 1.0

j_array = np.random.normal(mu, sigma, link_number)
print(j_array)

terms_array = []

f = open("SK_"+str(N)+".yaml", "a")
print("basis:", file=f)
print("  number_spins: "+str(N), file=f)
print("  hamming_weight: "+str(int(N/2)), file=f)
print("  spin_inversion: "+str(int(spin_inversion)), file=f)
print("  symmetries: []", file=f)
print("", file=f)
print("hamiltonian:", file=f)
print("  name: Sherrington-Kirkpatrick", file=f)

f.close()"""
