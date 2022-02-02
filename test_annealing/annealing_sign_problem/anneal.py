import argparse
#from .common import *
from annealing_sign_problem import * #extract_classical_ising_model
import ctypes
from collections import namedtuple
from typing import Tuple
import lattice_symmetries as ls
#import nqs_playground as nqs
#from nqs_playground.core import _get_dtype, _get_device
import torch
from torch import Tensor
#from torch.utils.tensorboard import SummaryWriter
from typing import Optional
import time
from loguru import logger
import sys
import numpy as np
import concurrent.futures

def project_dir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

"""class DenseModel(torch.nn.Module):
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
        return self.layers(x)"""

def _extract_classical_model_with_exact_fields(
    spins, hamiltonian, ground_state, sampled_power, device
):
    basis = hamiltonian.basis
    log_amplitude = ground_state.abs().log().unsqueeze(dim=1)
    log_sign = torch.where(
        ground_state >= 0,
        torch.scalar_tensor(0, device=device, dtype=ground_state.dtype),
        torch.scalar_tensor(np.pi, device=device, dtype=ground_state.dtype),
    ).unsqueeze(dim=1)

    @torch.no_grad()
    def log_coeff_fn(spin: Tensor) -> Tensor:
        spin = spin.cpu().numpy().view(np.uint64)
        if spin.ndim > 1:
            spin = spin[:, 0]
        indices = torch.from_numpy(ls.batched_index(basis, spin).view(np.int64)).to(device)
        a = log_amplitude[indices]
        b = log_sign[indices]
        return torch.complex(a, b)

    return extract_classical_ising_model(
        spins,
        hamiltonian,
        log_coeff_fn,
        sampled_power=sampled_power,
        device=device,
    )

def tune_sign_structure(
    spins: np.ndarray,
    hamiltonian: ls.Operator,
    log_psi: torch.nn.Module,
    number_sweeps: int,
    seed: Optional[int] = None,
    beta0: Optional[int] = None,
    beta1: Optional[int] = None,
    sampled_power: Optional[int] = None,
    device: Optional[torch.device] = None,
    scale_field=None,
    ground_state=None,
):
    spins0 = spins
    h, spins, x0, counts = extract_classical_ising_model(
        spins0,
        hamiltonian,
        log_psi,
        sampled_power=sampled_power,
        device=device,
        scale_field=scale_field,
    )
    x, _, e = sa.anneal(h, seed=seed, number_sweeps=number_sweeps, beta0=beta0, beta1=beta1)
    if ground_state is not None:
        h_exact, _, x0_exact, _ = _extract_classical_model_with_exact_fields(
            spins0,
            hamiltonian,
            ground_state,
            sampled_power=sampled_power,
            device=device,
        )
        # x_with_exact, _, e_with_exact = sa.anneal(
        #     h_exact, x0, seed=seed, number_sweeps=number_sweeps, beta0=beta0, beta1=beta1
        # )

    def extract_signs(bits):
        i = np.arange(spins.shape[0], dtype=np.uint64)
        signs = (bits[i // 64] >> (i % 64)) & 1
        signs = 1 - signs
        signs = torch.from_numpy(signs.view(np.int64))
        return signs

    signs = extract_signs(x)
    signs0 = extract_signs(x0)
    if ground_state is not None:
        signs0_exact = extract_signs(x0_exact)
        # signs_with_exact = extract_signs(x_with_exact)
        # accuracy_with_exact = torch.sum(signs0_exact == signs_with_exact).float() / spins.shape[0]
        accuracy_normal = torch.sum(signs0_exact == signs).float() / spins.shape[0]
        # logger.debug("SA accuracy with exact fields: {}", accuracy_with_exact)
        logger.debug("SA accuracy with approximate fields: {}", accuracy_normal)

        with open("SA_accuracy.txt", "a") as f2:
            np.savetxt(f2, [max(accuracy_normal, 1 - accuracy_normal)])
            f2.close()

    if False:  # NOTE: disabling for now
        if torch.sum(signs == signs0).float() / spins.shape[0] < 0.5:
            logger.warning("Applying global sign flip...")
            signs = 1 - signs
#    return spins, signs, counts
    return spins, signs0_exact, counts


Config = namedtuple(
    "Config",
    [
        "ground_state",
        "hamiltonian",
        "number_sa_sweeps",
        "number_monte_carlo_samples",
        "device",
        "output",
    ],
)

def make_log_coeff_fn(ground_state: np.ndarray, basis):
    log_amplitudes = ground_state.abs().log_().unsqueeze(dim=1)
    phases = torch.where(
        ground_state >= 0,
        torch.scalar_tensor(0.0, dtype=ground_state.dtype),
        torch.scalar_tensor(np.pi, dtype=ground_state.dtype),
    ).unsqueeze(dim=1)

    @torch.no_grad()
    def log_coeff_fn(spin: Tensor) -> Tensor:
        if not isinstance(spin, np.ndarray):
            spin = spin.cpu().numpy().view(np.uint64)
        if spin.ndim > 1:
            spin = spin[:, 0]
        indices = torch.from_numpy(ls.batched_index(basis, spin).view(np.int64))
        a = log_amplitudes[indices]
        b = phases[indices]
        return torch.complex(a, b)

    return log_coeff_fn

def _make_log_coeff_fn(amplitude, sign, basis, dtype, device):

    assert isinstance(amplitude, Tensor)
    assert isinstance(sign, torch.nn.Module)
    log_amplitude = amplitude.log().unsqueeze(dim=1)
    log_amplitude = log_amplitude.to(device=device, dtype=dtype)

    @torch.no_grad()
    def log_coeff_fn(spin: Tensor) -> Tensor:
        b = np.pi * sign(spin).argmax(dim=1, keepdim=True).to(dtype)
        spin = spin.cpu().numpy().view(np.uint64)
        if spin.ndim > 1:
            spin = spin[:, 0]
        indices = torch.from_numpy(ls.batched_index(basis, spin).view(np.int64)).to(device)
        a = log_amplitude[indices]
        return torch.complex(a, b)

    return log_coeff_fn

def anneal(config):
    hamiltonian = config.hamiltonian
    basis = hamiltonian.basis

#    try:
#        dtype = _get_dtype(config.model)
#    except AttributeError:
#        dtype = torch.float32
#    try:
#        device = _get_device(config.model)
#    except AttributeError:
#        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#    log_psi = _make_log_coeff_fn(config.ground_state.abs(), config.model, basis, dtype, device)
    log_psi = make_log_coeff_fn(config.ground_state, basis)
    sampled_power = 2.0  # True

    with torch.no_grad():
        p = config.ground_state.abs() ** sampled_power
        p = p.numpy()
        p /= np.sum(p)

    ground_state = config.ground_state.to(device)
    correct_sign_structure = torch.where(
        ground_state >= 0.0,
        torch.scalar_tensor(0, dtype=torch.int64, device=device),
        torch.scalar_tensor(1, dtype=torch.int64, device=device),
    )

    scale_field = [0] + [None for _ in range(1)]

    if sampled_power is not None:
        batch_indices = np.random.choice(
            basis.number_states, size=config.number_monte_carlo_samples, replace=True, p=p
        )
    else:
        batch_indices = np.random.choice(
            basis.number_states, size=config.number_monte_carlo_samples, replace=False, p=None
        )

    sampled_power = None  # True

    spins = basis.states#[batch_indices]
    spins, signs, counts = tune_sign_structure(
        spins,
        hamiltonian,
        log_psi,
        number_sweeps=config.number_sa_sweeps,
        sampled_power=sampled_power,
        device=device,
        scale_field=scale_field[0],
        ground_state=ground_state,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=int, default=10000, help="Number of SA sweeps.")
    parser.add_argument("--samples", type=int, default=30000, help="Number MC samples.")
    parser.add_argument("--seed", type=int, default=np.random.randint(100, 10000), help="Seed.")
    parser.add_argument("--device", type=str, default=None, help="Device.")
    args = parser.parse_args()

    with open("chain_length.txt", "a") as f5:
        np.savetxt(f5, [args.samples])
        f5.close()

#    ground_state, E, representatives = load_ground_state(
#        os.path.join(project_dir(), "../random_glass/SK_16.h5")
#    )
#    basis, hamiltonian = load_basis_and_hamiltonian(
#        os.path.join(project_dir(), "../random_glass/SK_16.yaml")
#    )

#    ground_state, E, representatives = load_ground_state(
#        os.path.join(project_dir(), "../kagome_16/tom_kagome_16.h5")
#    )
#    basis, hamiltonian = load_basis_and_hamiltonian(
#        os.path.join(project_dir(), "../kagome_16/tom_kagome_16.yaml")
#    )

    ground_state, E, representatives = load_ground_state(
        os.path.join(project_dir(), "../physical_systems/j1j2_square_4x4.h5")
    )
    basis, hamiltonian = load_basis_and_hamiltonian(
        os.path.join(project_dir(), "../physical_systems/j1j2_square_4x4.yaml")
    )

    basis.build(representatives)
    representatives = None
    logger.debug("Hilbert space dimension is {}", basis.number_states)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed + 1)
        logger.debug("Seeding PyTorch and NumPy with seed={}...", args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(device)

    config = Config(
        ground_state=ground_state,
        hamiltonian=hamiltonian,
        number_sa_sweeps=args.sweeps,
        number_monte_carlo_samples=args.samples,
        device=device,
        output="run/{}".format(args.seed if args.seed is not None else "dummy"),
    )

    os.makedirs(config.output, exist_ok=True)
    anneal(config)

if __name__ == "__main__":

    for iloop in range(10):
        main()
