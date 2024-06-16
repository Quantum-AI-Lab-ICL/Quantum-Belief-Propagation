import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy import linalg

from examples.example_utils import get_single_rho, matrix_3x3, \
    ham_setup, get_single_beliefs, \
    mean_norm, lat_ham_setup
from propagation import BeliefPropagator
from lattice_propagation import LatticeBeliefPropagator


def main():
    x_coeff = -2.5
    z_coeff = 0
    zz_coeff = 1.0
    size_1d = 9
    size_2d = 3

    error_1d = []
    error_2d = []
    beta_range = jnp.linspace(0.1, 1, 5)

    for beta in beta_range:
        H_lattice = matrix_3x3(x_coeff, zz_coeff)
        rho_lattice = linalg.expm(-beta * H_lattice)
        rho_lattice /= jnp.trace(rho_lattice)
        exact_results = get_single_beliefs(rho_lattice, size_1d)

        ham = ham_setup(size_1d, beta, x_coeff, z_coeff, zz_coeff)
        bp = BeliefPropagator(ham, 1)
        for i in range(size_1d):
            bp.step()
        results_1d = get_single_rho(bp.beliefs, size_1d)
        error_1d.append(mean_norm(results_1d - exact_results))

        lat_ham = lat_ham_setup(size_2d, beta, x_coeff, z_coeff, zz_coeff)
        lat_bp = LatticeBeliefPropagator(lat_ham, reg_factor=10e-6)
        for i in range(size_2d * size_2d):
            lat_bp.step()

        total_error = 0
        for r in range(size_2d):
            for c in range(size_2d):
                total_error += jnp.linalg.norm(
                    lat_bp.mean_single_belief(r, c) -
                    exact_results[r * size_2d + c]
                )
        error_2d.append(total_error / (size_2d * size_2d))

    plt.plot(beta_range, error_1d, label="Snake Approximation")
    plt.plot(beta_range, error_2d, label="2D Algorithm")
    plt.ylabel("""
               Mean matrix norm of difference between BP
               and exact results
               """)
    plt.xlabel("beta")
    plt.title("""
              Error against exact solution 2D lattice
              using snake approximation and 2D algorithm
              """)
    plt.legend()
    plt.tight_layout()
    plt.savefig("examples/results/2d/snake/comparison_algorithm.png")


if __name__ == "__main__":
    main()
