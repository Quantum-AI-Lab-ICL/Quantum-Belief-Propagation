import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy import linalg

from examples.example_utils import rdm, get_single_rho, matrix_3x3, \
    ham_setup, hamiltonian_matrix, get_diag_beliefs, get_single_beliefs, \
    mean_norm
from propagation import BeliefPropagator


def main():
    x_coeff = -2.5
    z_coeff = 0
    zz_coeff = 1.0
    size = 9

    chain_error = []
    lattice_error = []
    beta_range = jnp.linspace(0.1, 2, 10)

    for beta in beta_range:
        ham = ham_setup(size, beta, x_coeff, z_coeff, zz_coeff)
        bp = BeliefPropagator(ham, 1)
        for i in range(size):
            bp.step()
        bp_results = get_single_rho(bp.beliefs, size)

        H_chain = hamiltonian_matrix(ham)
        rho_chain = linalg.expm(H_chain)
        rho_chain /= jnp.trace(rho_chain)
        chain_results = get_single_beliefs(rho_chain, size)

        chain_error.append(mean_norm(bp_results - chain_results))

        H_lattice = matrix_3x3(x_coeff, zz_coeff)
        rho_lattice = linalg.expm(-beta * H_lattice)
        rho_lattice /= jnp.trace(rho_lattice)
        lattice_results = get_single_beliefs(rho_lattice, size)

        lattice_error.append(mean_norm(bp_results - lattice_results))

    plt.plot(beta_range, chain_error, label="Chain - 9")
    plt.plot(beta_range, lattice_error, label="Lattice - 3x3")
    plt.ylabel("""
               Mean matrix norm of difference between BP
               and exact results
               """)
    plt.xlabel("beta")
    plt.title("""
              Error against exact solution for 1D chain
              and snake approximation of 2D lattice
              """)
    plt.legend()
    plt.tight_layout()
    plt.savefig("examples/results/2d/snake/comparison_1d.png")


if __name__ == "__main__":
    main()
