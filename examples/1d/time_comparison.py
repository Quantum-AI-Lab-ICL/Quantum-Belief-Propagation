import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from jax.scipy import linalg

from propagation import BeliefPropagator
from examples.example_utils import get_single_rho, \
    ham_setup, get_diag_beliefs, hamiltonian_matrix


def main():
    beta = 1
    x_coeff = -1.05
    z_coeff = 0.5
    zz_coeff = 1.0

    size_range = range(3, 13)
    bp_times = []
    exact_times = []

    for size in size_range:
        ham = ham_setup(size, beta, x_coeff, z_coeff, zz_coeff)

        bp_start_time = time.perf_counter()
        bp = BeliefPropagator(ham, 1)
        for i in range(size):
            bp.step()
        get_single_rho(bp.beliefs, size)
        bp_end_time = time.perf_counter()

        exact_start_time = time.perf_counter()
        H = hamiltonian_matrix(ham)
        rho = linalg.expm(H)
        rho /= jnp.trace(rho)
        exact_beliefs = get_diag_beliefs(rho, size)
        get_single_rho(exact_beliefs, size)
        exact_end_time = time.perf_counter()

        bp_times.append(bp_end_time - bp_start_time)
        exact_times.append(exact_end_time - exact_start_time)

    plt.plot(size_range, bp_times, label="Belief Propagation")
    plt.plot(size_range, exact_times, label="Exact Diagonalisation")
    plt.ylabel("Computation time")
    plt.xlabel("Number of particles")
    plt.title("Computation time comparison for belief propagation and" +
              "exact diagonalisation")
    plt.legend()
    plt.savefig("examples/results/1d/time_comparison.png")


if __name__ == "__main__":
    main()
