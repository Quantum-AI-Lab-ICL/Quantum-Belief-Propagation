import jax.numpy as jnp
import jax.scipy.linalg as linalg
import jax.typing
import time

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
        hamiltonian = hamiltonian_setup(2, coef=-0.1)
        bp = BeliefPropagator(hamiltonian, 0)
        for i in range(5):
            bp.step()
        H = hamiltonian_matrix(hamiltonian)
        rho = linalg.expm(H)
        rho /= jnp.trace(rho)

    for size in range(3, 11):
        hamiltonian = hamiltonian_setup(size, coef=-0.1)
        print(f"Number of particles: {size}")

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

        print(total_error(bp_results, diag_results))
        print(bp_end_time - bp_start_time)
        print(diag_end_time - diag_start_time)


def get_bp_results(beliefs, size):
    results = jnp.zeros((size, MATRIX_SIZE_SINGLE, MATRIX_SIZE_SINGLE),
                        dtype=jnp.complex64)
    results = results.at[0].set(
        _double_to_single_trace(beliefs[0], 1))
    for i in range(beliefs.shape[0]):
        results = results.at[0].set(
            _double_to_single_trace(beliefs[i], 0))
    return results


def get_diag_results(rho, size):
    results = jnp.zeros((size, MATRIX_SIZE_SINGLE, MATRIX_SIZE_SINGLE),
                        dtype=jnp.complex64)
    for i in range(size):
        results = results.at[0].set(rdm(rho, partial_dim=1, pos=i))
    return results


def hamiltonian_setup(size: int, coef: int) -> Hamiltonian:
    """
    TODO
    """

    ham = Hamiltonian(size)
    for i in range(size):
        ham.set_param_single(i, Pauli.X, coef)
    for i in range(size - 1):
        ham.set_param_double(i, Pauli.Z, Pauli.Z, coef)
    ham.compute_partial_hamiltonians()
    return ham


def total_error(bp_results: jax.typing.ArrayLike,
                diag_results: jax.typing.ArrayLike) -> jnp.float64:
    """
    TODO
    """

    result = 0
    for i in range(bp_results.shape[0]):
        result += jnp.linalg.norm(bp_results[i] - diag_results[i])
    return result


if __name__ == "__main__":
    main()
