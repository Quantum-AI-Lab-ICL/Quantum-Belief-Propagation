import jax.numpy as jnp
import jax.scipy.linalg as linalg
import jax.typing
import time
from jax import random

import matplotlib.pyplot as plt

from const import MATRIX_SIZE_SINGLE, MATRIX_SIZE_DOUBLE
from hamiltonian import Hamiltonian
from pauli import Pauli
from propagation import BeliefPropagator
from examples.example_utils import hamiltonian_matrix, rdm, \
    get_single_rho, trans_mag, correlation


def main():
    jnp.set_printoptions(precision=7, suppress=True, linewidth=1000)
    beta = 1

    # Warmup
    for _ in range(10):
        hamiltonian = hamiltonian_setup(2, beta=beta, coef=1, seed=0)
        bp = BeliefPropagator(hamiltonian, 1)
        for i in range(5):
            bp.step()
        H = hamiltonian_matrix(hamiltonian)
        rho = linalg.expm(H)
        rho /= jnp.trace(rho)

        trans_mag_bp = []
        trans_mag_diag = []
        correlation_bp = []
        correlation_diag = []
        density = []

    for size in range(3, 5):
        print(f"Number of particles: {size}")
        num_experiments = 1
        total_error = 0

        for seed in range(num_experiments):
            hamiltonian = hamiltonian_setup(size, beta, coef=1, seed=seed)

            # Belief propagatioin
            bp_start_time = time.perf_counter()
            bp = BeliefPropagator(hamiltonian, 0)
            for i in range(size):
                bp.step()
            bp_results = get_single_rho(bp.beliefs, size)
            bp_end_time = time.perf_counter()

            # Exact diagonalisation
            diag_start_time = time.perf_counter()
            H = hamiltonian_matrix(hamiltonian)
            rho = linalg.expm(H)
            rho /= jnp.trace(rho)
            diag_beliefs = get_diag_beliefs(rho, size)
            diag_results = get_single_rho(diag_beliefs, size)
            diag_end_time = time.perf_counter()

            trans_mag_bp.append(trans_mag(bp_results))
            trans_mag_diag.append(trans_mag(diag_results))
            correlation_bp.append(correlation(bp.beliefs))
            correlation_diag.append(correlation(diag_beliefs))
            density.append(jnp.trace(rho @ H / (-beta)) / size)

            total_error += result_error(bp_results, diag_results)
            if seed == 0:
                print("BP time:", bp_end_time - bp_start_time)
                print("Exact time", diag_end_time - diag_start_time)

        print("Average error:", total_error / num_experiments)

    plt.plot(density, trans_mag_diag)
    plt.scatter(density, trans_mag_bp)
    plt.ylabel("M_x")
    plt.xlabel("<H>/N")
    plt.show()


def get_diag_results(rho, size):
    results = jnp.zeros((size, MATRIX_SIZE_SINGLE, MATRIX_SIZE_SINGLE),
                        dtype=jnp.complex64)
    for i in range(size):
        results = results.at[i].set(rdm(rho, partial_dim=1, pos=i))
    return results


def get_diag_beliefs(rho, size):
    results = jnp.zeros((size - 1, MATRIX_SIZE_DOUBLE, MATRIX_SIZE_DOUBLE),
                        dtype=jnp.complex64)
    for i in range(size - 1):
        results = results.at[i].set(rdm(rho, partial_dim=2, pos=i))
    return results


def hamiltonian_setup(size: jnp.int32, beta: jnp.float32,
                      coef: jnp.float32, seed: jnp.int32) -> Hamiltonian:
    """
    TODO
    """

    # key = random.key(seed)
    # num_values = 2 * size - 1
    # coef_mod = random.normal(key, (2 * num_values,))
    ham = Hamiltonian(size, beta)
    for i in range(size):
        ham.set_param_single(i, Pauli.X, -1.05)
        ham.set_param_single(i, Pauli.Z, 0.5)
    for i in range(size - 1):
        ham.set_param_double(i, Pauli.Z, Pauli.Z, coef)
    ham.compute_partial_hamiltonians()
    return ham


def result_error(bp_results: jax.typing.ArrayLike,
                 diag_results: jax.typing.ArrayLike) -> jnp.float64:
    """
    TODO
    """

    result = 0
    for i in range(bp_results.shape[0]):
        result += jnp.linalg.norm(bp_results[i] - diag_results[i])
    return result / bp_results.shape[0]


if __name__ == "__main__":
    main()
