import lattice_symmetries as ls
import numpy as np
import h5py

np.random.seed(167453)

N = 14
spin_inversion = 1

link_number = int(N*(N-1)/2)

# Generate random couplings

mu = 0.0
sigma = 1.0

f = open("SK_"+str(N)+".yaml", "a")
print("basis:", file=f)
print("  number_spins: "+str(N), file=f)
print("  hamming_weight: "+str(int(N/2)), file=f)
#print("  spin_inversion: "+str(int(spin_inversion)), file=f)
print("  symmetries: []", file=f)
print("", file=f)
print("hamiltonian:", file=f)
print("  name: Sherrington-Kirkpatrick", file=f)
print("  terms:", file=f)

for iloop in range(N-1):
    for jloop in range(iloop+1, N):

        coupling = np.random.normal(mu, sigma)/100.
        print(coupling)

        print("    - matrix: [["+str(coupling)+", 0, 0, 0],", file=f)
        print("               [0, "+str(-coupling)+", "+str(2*coupling)+", 0],", file=f)
        print("               [0, "+str(2*coupling)+", "+str(-coupling)+", 0],", file=f)
        print("               [0, 0, 0, "+str(coupling)+"]]", file=f)
        print("", file=f)
        print("      sites: [["+str(iloop)+", "+str(jloop)+"]]", file=f)
        print("", file=f)

print("observables: []", file=f)
print("number_vectors: 1", file=f)
print('output: "SK_'+str(N)+'.h5"', file=f)
print('datatype: "float32"', file=f)
print('max_primme_block_size: 4', file=f)
print('max_primme_basis_size: 20', file=f)

f.close()
