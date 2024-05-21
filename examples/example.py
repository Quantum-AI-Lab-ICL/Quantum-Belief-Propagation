import jax.numpy as jnp
import jax.scipy.linalg as linalg
import jax.typing
import time
from jax import random

from const import MATRIX_SIZE_SINGLE
from hamiltonian import Hamiltonian
from pauli import Pauli
from propagation import BeliefPropagator
from utils import _double_to_single_trace
from examples.example_utils import hamiltonian_matrix, rdm


def main():
    jnp.set_printoptions(precision=7, suppress=True, linewidth=1000)

    # Warmup
    for _ in range(10):
        hamiltonian = hamiltonian_setup(2, coef=1, seed=0)
        bp = BeliefPropagator(hamiltonian, 1)
        for i in range(5):
            bp.step()
        H = hamiltonian_matrix(hamiltonian)
        rho = linalg.expm(H)
        rho /= jnp.trace(rho)

    for size in range(3, 9):
        print(f"Number of particles: {size}")
        num_experiments = 1
        total_error = 0

        for seed in range(num_experiments):
            hamiltonian = hamiltonian_setup(size, coef=1, seed=seed)

            # Belief propagatioin
            bp_start_time = time.perf_counter()
            bp = BeliefPropagator(hamiltonian, 0)
            for i in range(size):
                bp.step()
            bp_results = get_bp_results(bp.beliefs, size)
            bp_end_time = time.perf_counter()

            # Exact diagonalisation
            diag_start_time = time.perf_counter()
            H = hamiltonian_matrix(hamiltonian)
            rho = linalg.expm(H)
            rho /= jnp.trace(rho)
            diag_results = get_diag_results(rho, size)
            diag_end_time = time.perf_counter()

            total_error += result_error(bp_results, diag_results)
            if seed == 0:
                print("BP time:", bp_end_time - bp_start_time)
                print("Exact time", diag_end_time - diag_start_time)

        print("Average error:", total_error / num_experiments)


def get_bp_results(beliefs, size):
    results = jnp.zeros((size, MATRIX_SIZE_SINGLE, MATRIX_SIZE_SINGLE),
                        dtype=jnp.complex64)
    results = results.at[0].set(
        _double_to_single_trace(beliefs[0], 1))
    for i in range(beliefs.shape[0]):
        results = results.at[i+1].set(
            _double_to_single_trace(beliefs[i], 0))
    return results


def get_diag_results(rho, size):
    results = jnp.zeros((size, MATRIX_SIZE_SINGLE, MATRIX_SIZE_SINGLE),
                        dtype=jnp.complex64)
    for i in range(size):
        results = results.at[i].set(rdm(rho, partial_dim=1, pos=i))
    return results


def hamiltonian_setup(size: jnp.int32, coef: jnp.complex64,
                      seed: jnp.int32) -> Hamiltonian:
    """
    TODO
    """

    # key = random.key(seed)
    # num_values = 2 * size - 1
    # coef_mod = random.normal(key, (2 * num_values,))
    ham = Hamiltonian(size, beta=1)
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
