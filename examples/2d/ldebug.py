import jax.numpy as jnp
from jax.scipy import linalg
import matplotlib.pyplot as plt
from jax import random

from lattice_hamiltonian import LatticeHamiltonian
from lattice_propagation import LatticeBeliefPropagator
from pauli import Pauli
from utils import _random_normalised_hermitian

from examples.example_utils import rdm, matrix_3x3, get_single_rho


def main():
    x_coef = -3
    zz_coef = -1
    size = 3

    errors = []

    beta = 2.0
    space = jnp.linspace(0.01, 0.1, 10)
    for reg_factor in space:
        lat_ham = LatticeHamiltonian(3, 3, beta)
        for r in range(size):
            for c in range(size):
                lat_ham.set_param_single(r, c, Pauli.X, x_coef)
        for r in range(size):
            for c in range(size - 1):
                lat_ham.set_param_double_row(r, c, Pauli.Z, Pauli.Z, zz_coef)
        for c in range(size):
            for r in range(size - 1):
                lat_ham.set_param_double_col(c, r, Pauli.Z, Pauli.Z, zz_coef)
        lat_ham.compute_partial_hamiltonians()

        H = matrix_3x3(x_coef, zz_coef)
        rho = linalg.expm(-beta * H)
        rho /= jnp.trace(rho)

        exact_sol = jnp.zeros((size, size, 2, 2), dtype=jnp.complex64)
        for r in range(size):
            for c in range(size):
                exact_sol = exact_sol.at[r, c].set(rdm(rho, 1, r * size + c))

        propagator = LatticeBeliefPropagator(lat_ham, reg_factor)
        for i in range(size * size):
            propagator.step()
            total_error = 0.0
            for r in range(size):
                for c in range(size):
                    total_error += jnp.linalg.norm(
                        propagator.mean_single_belief(r, c) -
                        exact_sol[r, c]
                    )
        print(total_error / (size * size))
        errors.append(total_error / (size * size))

    plt.plot(space, errors)
    plt.xlabel("Regularisation factor")
    plt.ylabel("Average norm of error")
    plt.title("Error against exact solution in 3x3 matrices by " +
              "regularisation factor")
    plt.savefig("examples/results/error_3x3_by_reg_scalar.png")


if __name__ == "__main__":
    main()
