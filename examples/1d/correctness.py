from jax.scipy import linalg
import jax.numpy as jnp
import matplotlib.pyplot as plt

from examples.example_utils import get_single_rho, \
    ham_setup, get_diag_beliefs, hamiltonian_matrix
from propagation import BeliefPropagator


def main():
    beta = 1
    x_coeff = -1.05
    z_coeff = 0.5
    zz_coeff = 1.0

    size_range = range(3, 13)
    error = []

    for size in size_range:
        ham = ham_setup(size, beta, x_coeff, z_coeff, zz_coeff)
        bp = BeliefPropagator(ham)
        for i in range(size):
            bp.step()
        bp_results = get_single_rho(bp.beliefs, size)

        H = hamiltonian_matrix(ham)
        rho = linalg.expm(H)
        rho /= jnp.trace(rho)
        exact_beliefs = get_diag_beliefs(rho, size)
        exact_results = get_single_rho(exact_beliefs, size)

        error.append(jnp.linalg.norm(
            jnp.mean(bp_results - exact_results, axis=0)
        ) / size)

    plt.plot(size_range, error)
    plt.ylabel("""Mean matrix norm of difference between BP and
               exact results""")
    plt.xlabel("Number of particles")
    plt.title("""Error against exact solution in 1D systems by
              system size""")
    plt.tight_layout()
    plt.savefig("examples/results/1d/correctness.png")


if __name__ == "__main__":
    main()
