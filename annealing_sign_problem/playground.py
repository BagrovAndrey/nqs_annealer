import numpy as np
import torch
import yaml
import nqs_playground as nqs
import nqs_playground.sgd as sgd
import nqs_playground.core as core
import lattice_symmetries as ls
import h5py
import unpack_bits

from collections import namedtuple
import time
import os
from loguru import logger
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

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

def combine_amplitude_and_sign(*modules, use_jit: bool = True
 ) -> torch.nn.Module:
     r"""Combines two torch.nn.Modules representing amplitudes and signs of the
     wavefunction coefficients into one model.
     """

     class CombiningState(torch.nn.Module):

         def __init__(self, amplitude, phase):
             super().__init__()
             self.amplitude = amplitude
             self.phase = phase

         def forward(self, x):

             a = self.amplitude(x)
             b = (-(self.phase(x)-1.)*1.570796326794897).view([-1, 1])
             cat = torch.cat([a, b], dim=1)
             complexify = torch.view_as_complex(cat).view([-1,1])

             return complexify

     m = CombiningState(*modules)
#     if use_jit:
#         m = torch.jit.script(m)
     return m

def combine_amplitude_and_explicit_sign(
     *modules, use_jit: bool = True
 ) -> torch.nn.Module:
     r"""Combines two torch.nn.Modules representing amplitudes and signs of the
     wavefunction coefficients into one model.
     """

     class CombiningState(torch.nn.Module):

         def __init__(self, amplitude, exact_sign):
             super().__init__()
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

Config = namedtuple(
    "Config",
    [
        "amplitude",
        "phase",
        "hamiltonian",
        "output",
        "epochs",
        "sampling_options",
        "optimizer",
        "scheduler",
        "exact",
        "constraints",
        "inference_batch_size",
        "basis",
        "ground_state"
    ],
    defaults=[],
)


def _should_optimize(module):
    return any(map(lambda p: p.requires_grad, module.parameters()))

class Runner(nqs.RunnerBase):
    def __init__(self, config):
        super().__init__(config)
#        self.combined_state = combine_amplitude_and_explicit_sign(
#            self.config.amplitude, self.config.phase, use_jit=False
#        )
        self.combined_state = combine_amplitude_and_sign(
            self.config.amplitude, self.config.phase, use_jit=False
        )

    def inner_iteration(self, states, log_probs, weights):
        assert weights.dtype == torch.float64
        assert log_probs.dtype == torch.float64
        logger.info("Inner iteration...")
        tick = time.time()
        E = self.compute_local_energies(states, weights)
        E = E.real.to(dtype=weights.dtype)
        states = states.view(-1, states.size(-1))
        log_probs = states.view(-1)
        weights = weights.view(-1)

        # Compute output gradient
        with torch.no_grad():
            grad = E - torch.dot(E, weights)
            grad *= 2 * weights
            grad = grad.view(-1, 1)
            grad_norm = torch.linalg.norm(grad)
            logger.info("‖∇E‖₂ = {}", grad_norm)
            self.tb_writer.add_scalar("loss/grad", grad_norm, self.global_index)

        self.config.optimizer.zero_grad()
        batch_size = self.config.inference_batch_size

        # Compute overlap
        with torch.no_grad():

            all_spins = torch.from_numpy(self.config.basis.states.view(np.int64)).reshape(-1,1)
            print("all_spins", all_spins)
            neural_state = self.combined_state(all_spins)
            print("neural_state", neural_state)
            norm = torch.sqrt(torch.dot(torch.conj(torch.exp(neural_state.reshape(1,-1)[0])), torch.exp(neural_state.reshape(1,-1)[0]))).real
            print("norm", norm)
            neural_state=torch.exp(neural_state.reshape(1,-1)[0])/norm
            print("neural_state", neural_state)
            print(torch.dot(torch.conj(neural_state), neural_state))
            ground_state = self.config.ground_state
            print("ground_state_norm", torch.dot(torch.conj(ground_state), ground_state))
            overlap = torch.dot(torch.conj(ground_state.type(torch.complex64)), neural_state.type(torch.complex64))
            print("overlap", overlap, torch.abs(overlap))

        # Computing gradients for the amplitude network
        logger.info("Computing gradients...")
        if _should_optimize(self.config.amplitude):
            self.config.amplitude.train()
            forward_fn = self.amplitude_forward_fn
            for (states_chunk, grad_chunk) in core.split_into_batches((states, grad), batch_size):
                output = forward_fn(states_chunk)
                output.backward(grad_chunk) # , retain_graph=True)

        # Computing gradients for the phase network
        if _should_optimize(self.config.phase):
            self.config.phase.train()
            forward_fn = self.config.phase
            for (states_chunk, grad_chunk) in core.split_into_batches((states, grad), batch_size):
                output = forward_fn(states_chunk)
                output.backward(grad_chunk)

        self.config.optimizer.step()
        if self.config.scheduler is not None:
            self.config.scheduler.step()
        self.checkpoint(init={"optimizer": self.config.optimizer.state_dict()})
        tock = time.time()
        logger.info("Completed inner iteration in {:.1f} seconds!", tock - tick)
          
def main():
    number_spins = 32
    device = torch.device("cpu")

    basis, hamiltonian = load_basis_and_hamiltonian("pyrochlore.yaml")
    ground_state, E, representatives = load_ground_state("pyrochlore.h5")
#    basis, hamiltonian = load_basis_and_hamiltonian("kagome_12_periodic.yaml")
#    ground_state, E, representatives = load_ground_state("kagome_12_periodic.h5")

#    basis, hamiltonian = load_basis_and_hamiltonian("square_24_periodic.yaml")
#    ground_state, E, representatives = load_ground_state("square_24_periodic.h5")
    sign_structure = torch.sign(ground_state)
    torch_representatives = torch.from_numpy(representatives.view(np.int64)).reshape(-1,1)
    basis.build()

    getting_sign = vector_to_module(sign_structure, basis)

    amplitude = torch.nn.Sequential(
        nqs.Unpack(number_spins),
        torch.nn.Linear(number_spins, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1, bias=False),
    ).to(device)

    neural_sign = torch.nn.Sequential(
        nqs.Unpack(number_spins),
        torch.nn.Linear(number_spins, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1, bias=False),
    ).to(device)

    combined_state = combine_amplitude_and_explicit_sign(amplitude, getting_sign)
#    combined_state = combine_amplitude_and_sign(amplitude, neural_sign)

    optimizer = torch.optim.SGD(
        list(combined_state.parameters()),
        lr=1.4e-2,
        momentum=0.8,
#        weight_decay=1e-4,
    )
    optimizer = torch.optim.Adam(list(combined_state.parameters()), lr=1e-2)
    options = Config(
        amplitude=amplitude,
        phase=getting_sign,
#        phase=neural_sign,
        hamiltonian=hamiltonian,
        output="test.result",
        epochs=500,
        sampling_options=nqs.SamplingOptions(number_samples=10000, number_chains=10, mode='exact'),
        exact=None,  # ground_state[:, 0],
        constraints=None,#{"hamming_weight": lambda i: 0.1},
        optimizer=optimizer,
        scheduler=None,
        inference_batch_size=8192,
        basis=basis,
        ground_state=ground_state
    )

    runner = Runner(options)
    runner.run(1)

main()
