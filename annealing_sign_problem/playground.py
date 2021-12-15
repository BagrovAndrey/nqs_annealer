import numpy as np
import torch
import yaml
import nqs_playground as nqs
import lattice_symmetries as ls
import h5py

class DenseModel(torch.nn.Module):
    def __init__(self, shape, number_features, use_batchnorm=True, dropout=None):
        super().__init__()
        number_blocks = len(number_features)
        number_features = number_features + [2]
        layers = [torch.nn.Linear(shape[0] * shape[1], number_features[0])]
        for i in range(1, len(number_features)):
            layers.append(torch.nn.ReLU(inplace=True))
            if use_batchnorm:
                layers.append(torch.nn.BatchNorm1d(number_features[i - 1]))
            if dropout is not None:
                layers.append(torch.nn.Dropout(p=dropout, inplace=True))
            layers.append(torch.nn.Linear(number_features[i - 1], number_features[i]))

        self.shape = shape
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        number_spins = self.shape[0] * self.shape[1]
        x = nqs.unpack(x, number_spins)
        return self.layers(x)

def load_ground_state(filename: str):
    with h5py.File(filename, "r") as f:
        ground_state = f["/hamiltonian/eigenvectors"][:]
        ground_state = ground_state.squeeze()
        energy = f["/hamiltonian/eigenvalues"][0]
        basis_representatives = f["/basis/representatives"][:]
    return torch.from_numpy(ground_state), energy, basis_representatives


def load_basis_and_hamiltonian(filename: str):
    with open(filename, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    basis = ls.SpinBasis.load_from_yaml(config["basis"])
    hamiltonian = ls.Operator.load_from_yaml(config["hamiltonian"], basis)
    return basis, hamiltonian

ground_state, E, representatives = load_ground_state("pyrochlore.h5")
basis, hamiltonian = load_basis_and_hamiltonian("pyrochlore.yaml")

sign_structure = torch.sign(ground_state)
