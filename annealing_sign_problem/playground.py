import numpy as np
import torch
import yaml
import nqs_playground as nqs
import lattice_symmetries as ls
import h5py
import unpack_bits

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

def combine_amplitude_and_sign(
     *modules, apply_log: bool = False, out_dim: int = 1, use_jit: bool = True
 ) -> torch.nn.Module:
     r"""Combines two torch.nn.Modules representing amplitudes and signs of the
     wavefunction coefficients into one model.
     """
     if out_dim != 1 and out_dim != 2:
         raise ValueError("invalid out_dim: {}; expected either 1 or 2".format(out_dim))
     if out_dim == 1 and apply_log:
         raise ValueError("apply_log is incompatible with out_dim=1")

     class CombiningState(torch.nn.Module):
         __constants__ = ["apply_log", "out_dim"]

         def __init__(self, amplitude, phase):
             super().__init__()
             self.apply_log = apply_log
             self.out_dim = out_dim
             self.amplitude = amplitude
             self.phase = phase

         def forward(self, x):
             a = torch.log(self.amplitude(x)) if self.apply_log else self.amplitude(x)
             if self.out_dim == 1:
                 b = (1 - 2 * torch.argmax(self.phase(x), dim=1)).to(torch.float32).view([-1, 1])
                 a *= b
                 return a
             else:
                 b = 3.141592653589793 * torch.argmax(self.phase(x), dim=1).to(torch.float32).view(
                     [-1, 1]
                 )
                 return torch.cat([a, b], dim=1)

     m = CombiningState(*modules)
     if use_jit:
         m = torch.jit.script(m)
     return m

def combine_amplitude_and_explicit_sign(
     *modules, apply_log: bool = False, out_dim: int = 1, use_jit: bool = True
 ) -> torch.nn.Module:
     r"""Combines two torch.nn.Modules representing amplitudes and signs of the
     wavefunction coefficients into one model.
     """
     if out_dim != 1 and out_dim != 2:
         raise ValueError("invalid out_dim: {}; expected either 1 or 2".format(out_dim))
     if out_dim == 1 and apply_log:
         raise ValueError("apply_log is incompatible with out_dim=1")

     class CombiningState(torch.nn.Module):
         __constants__ = ["apply_log", "out_dim"]

         def __init__(self, amplitude, exact_sign):
             super().__init__()
             self.apply_log = apply_log
             self.out_dim = out_dim
             self.amplitude = amplitude
             self.exact_sign = exact_sign

         def forward(self, x):
             a = self.amplitude(x)
             b = (-(self.exact_sign(x)-1.)*1.570796326794897).view([-1, 1])

             cat = torch.cat([a, b], dim=1)
             complexify = torch.view_as_complex(cat).view([-1,1])

             return complexify

     m = CombiningState(*modules)
 #    if use_jit:
 #        m = torch.jit.script(m)
     return m

def vector_to_module(vector: torch.Tensor, basis) -> torch.nn.Module:

    class SignModule(torch.nn.Module):

        def __init__(self, vector, basis):
            super().__init__()
            self.vector = vector
            self.basis = basis

        def forward(self, x):
            x = x[:,:1].reshape(1,-1)[0].cpu().detach().numpy().view(np.uint64)
            ind = self.basis.batched_index(x)
            ind = torch.from_numpy(ind.view(np.int64))
            sign = self.vector[ind]
            return sign

    sign_module = SignModule(vector, basis)
    return sign_module
          
def main():
    number_spins = 32
    device = torch.device("cpu")

    basis, hamiltonian = load_basis_and_hamiltonian("pyrochlore.yaml")
    ground_state, E, representatives = load_ground_state("pyrochlore.h5")
    sign_structure = torch.sign(ground_state)
    basis.build()

#   Testing data transformations

    test_spin = np.array([representatives[123], representatives[125], representatives[323], representatives[1002], representatives[1311]])
    recovered_inds = basis.batched_index(test_spin)

    torch_spins = torch.from_numpy(test_spin.view(np.int64)).reshape(-1,1)
    np_spins = torch_spins[:,:1].reshape(1,-1)[0].numpy().view(np.uint64)

    getting_sign = vector_to_module(sign_structure, basis)
    print("Signs harvested", getting_sign(torch_spins))    

    amplitude = torch.nn.Sequential(
        nqs.Unpack(number_spins),
        torch.nn.Linear(number_spins, 144),
        torch.nn.ReLU(),
        torch.nn.Linear(144, 1, bias=False),
    ).to(device)

    combined_state = combine_amplitude_and_explicit_sign(amplitude, getting_sign)
#    print(combined_state(torch_spins))

    local_en = nqs.local_values(torch_spins, hamiltonian, combined_state)
    print(local_en)

main()
