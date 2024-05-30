import jax.numpy as jnp
from jax.scipy import linalg

from grid_hamiltonian import GridHamiltonian
from pauli import Pauli
from grid_propagation import GridBeliefPropagator
from utils import _double_to_single_trace

from examples.example_utils import rdm, matrix_3x3, get_single_rho


if __name__ == "__main__":
    x_coef = -3
    zz_coef = 1
    beta = 1

    grid_ham = GridHamiltonian(3, 3, 1)
    for r in range(3):
        for c in range(3):
            grid_ham.set_param_single(r, c, Pauli.X, x_coef)
    for r in range(3):
        for c in range(2):
            grid_ham.set_param_double_row(r, c, Pauli.Z, Pauli.Z, zz_coef)
    for c in range(3):
        for r in range(2):
            grid_ham.set_param_double_col(c, r, Pauli.Z, Pauli.Z, zz_coef)
    grid_ham.compute_partial_hamiltonians()

    propagator = GridBeliefPropagator(grid_ham)
    for _ in range(18):
        propagator.step()

    b00 = _double_to_single_trace(propagator.beliefs_row[0, 0], 1)
    print(_double_to_single_trace(propagator.beliefs_row[0, 0], 1))
    print(_double_to_single_trace(propagator.beliefs_row[0, 0], 0))
    print(_double_to_single_trace(propagator.beliefs_row[1, 1], 1))
    print(_double_to_single_trace(propagator.beliefs_row[1, 1], 0))
    print(b00)

    H = matrix_3x3(x_coef, zz_coef)
    rho = linalg.expm(-H)
    rho /= jnp.trace(rho)
    print(jnp.linalg.norm(b00 - rdm(rho, 1, 0)))
