import jax.numpy as jnp
from jax.scipy import linalg
import matplotlib.pyplot as plt

from examples.example_utils import get_single_rho, matrix_3x3, \
    ham_setup, get_single_beliefs, \
    mean_norm, lat_ham_setup
from lattice_propagation import LatticeBeliefPropagator


if __name__ == "__main__":

    x_coeff = -2.5
    z_coeff = 0
    zz_coeff = 1.0
    size = 3

    error = []
    beta = 0.5

    H_lattice = matrix_3x3(x_coeff, zz_coeff)
    rho_lattice = linalg.expm(-beta * H_lattice)
    rho_lattice /= jnp.trace(rho_lattice)
    exact_results = get_single_beliefs(rho_lattice, size * size)

    lat_ham = lat_ham_setup(size, beta, x_coeff, z_coeff, zz_coeff)
    lat_bp = LatticeBeliefPropagator(lat_ham, reg_factor=10e-6)
    step_range = range(2 * size * size)
    for i in step_range:
        lat_bp.step()
        total_error = 0
        for r in range(size):
            for c in range(size):
                total_error += jnp.linalg.norm(
                    lat_bp.mean_single_belief(r, c) -
                    exact_results[r * size + c]
                )
        error.append(total_error / (size * size))

    plt.plot(step_range, error)
    plt.ylabel("""
               Mean matrix norm of difference between BP
               and exact results
               """)
    plt.xlabel("Number of steps")
    plt.title("""
              Error against exact solution in 2D systems by
              number of steps
              """)
    plt.tight_layout()
    plt.savefig("examples/results/2d/convergence.png")
