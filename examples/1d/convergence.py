import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy import linalg

from propagation import BeliefPropagator
from examples.example_utils import get_single_rho, \
    ham_setup, get_diag_beliefs, hamiltonian_matrix


def main():
    size = 10
    beta = 1
    x_coeff = -1.05
    z_coeff = 0.5
    zz_coeff = 1.0

    error = []

    ham = ham_setup(size, beta, x_coeff, z_coeff, zz_coeff)
    H = hamiltonian_matrix(ham)
    rho = linalg.expm(H)
    rho /= jnp.trace(rho)
    exact_beliefs = get_diag_beliefs(rho, size)
    exact_results = get_single_rho(exact_beliefs, size)

    bp = BeliefPropagator(ham, 1)
    step_range = range(size * 2)
    for i in step_range:
        bp.step()
        bp_results = get_single_rho(bp.beliefs, size)
        error.append(jnp.linalg.norm(
            jnp.mean(bp_results - exact_results, axis=0)
        ) / size)

    plt.plot(step_range, error)
    plt.ylabel("Mean matrix norm of difference between BP and exact " +
               "results")
    plt.xlabel("Number of steps")
    plt.title("Error against exact solution in 1D systems by " +
              "number of steps")
    plt.savefig("examples/results/1d/convergence.png")


if __name__ == "__main__":
    main()
