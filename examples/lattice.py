import jax.numpy as jnp
from jax.scipy import linalg

from lattice_hamiltonian import LatticeHamiltonian
from lattice_propagation import LatticeBeliefPropagator
from pauli import Pauli
from utils import _double_to_single_trace

from examples.example_utils import rdm, matrix_3x3, get_single_rho


if __name__ == "__main__":
    x_coef = -3
    zz_coef = -1
    beta = 0.4

    size = 3

    lat_ham = LatticeHamiltonian(3, 3, beta)
    for r in range(3):
        for c in range(3):
            lat_ham.set_param_single(r, c, Pauli.X, x_coef)
    for r in range(3):
        for c in range(2):
            lat_ham.set_param_double_row(r, c, Pauli.Z, Pauli.Z, zz_coef)
    for c in range(3):
        for r in range(2):
            lat_ham.set_param_double_col(c, r, Pauli.Z, Pauli.Z, zz_coef)
    lat_ham.compute_partial_hamiltonians()

    H = matrix_3x3(x_coef, zz_coef)
    rho = linalg.expm(-beta * H)
    rho /= jnp.trace(rho)

    exact_sol = jnp.zeros((size, size, 2, 2), dtype=jnp.complex64)
    for r in range(size):
        for c in range(size):
            exact_sol = exact_sol.at[r, c].set(rdm(rho, 1, r * size + c))

    propagator = LatticeBeliefPropagator(lat_ham)
    for i in range(18):
        propagator.step()
        print("step", i)
        total_error = 0.0
        for r in range(size):
            for c in range(size):
                total_error += jnp.linalg.norm(
                    propagator.mean_single_belief(r, c) -
                    exact_sol[r, c]
                )
        print(total_error / (size * size))
