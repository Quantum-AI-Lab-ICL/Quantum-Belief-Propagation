import jax.numpy as jnp
from jax.scipy import linalg

from lattice_hamiltonian import LatticeHamiltonian
from lattice_propagation import LatticeBeliefPropagator
from pauli import Pauli
from utils import _double_to_single_trace

from examples.example_utils import rdm, matrix_3x3, get_single_rho


if __name__ == "__main__":
    x_coef = -3
    zz_coef = 1
    beta = 1

    lat_ham = LatticeHamiltonian(3, 3, 1)
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

    propagator = LatticeBeliefPropagator(lat_ham)
    for i in range(9):
        print("step", i)
        propagator.step()

    b00 = _double_to_single_trace(propagator.beliefs_row[0, 0], 1)
    b01 = _double_to_single_trace(propagator.beliefs_row[0, 1], 1)
    b02 = _double_to_single_trace(propagator.beliefs_row[0, 1], 0)
    b10 = _double_to_single_trace(propagator.beliefs_row[1, 0], 1)
    b11 = _double_to_single_trace(propagator.beliefs_row[1, 1], 1)
    b12 = _double_to_single_trace(propagator.beliefs_row[1, 1], 0)
    b20 = _double_to_single_trace(propagator.beliefs_row[2, 0], 1)
    b21 = _double_to_single_trace(propagator.beliefs_row[2, 1], 1)
    b22 = _double_to_single_trace(propagator.beliefs_row[2, 1], 0)
    # print(_double_to_single_trace(propagator.beliefs_row[0, 0], 1))
    # print(_double_to_single_trace(propagator.beliefs_row[0, 0], 0))
    # print(_double_to_single_trace(propagator.beliefs_row[1, 1], 1))
    # print(_double_to_single_trace(propagator.beliefs_row[1, 1], 0))

    H = matrix_3x3(x_coef, zz_coef)
    rho = linalg.expm(-H)
    rho /= jnp.trace(rho)
    print(jnp.linalg.norm(b00 - rdm(rho, 1, 0)))
    print(jnp.linalg.norm(b01 - rdm(rho, 1, 1)))
    print(jnp.linalg.norm(b02 - rdm(rho, 1, 2)))
    print(jnp.linalg.norm(b10 - rdm(rho, 1, 3)))
    print(jnp.linalg.norm(b11 - rdm(rho, 1, 4)))
    print(jnp.linalg.norm(b12 - rdm(rho, 1, 5)))
    print(jnp.linalg.norm(b20 - rdm(rho, 1, 6)))
    print(jnp.linalg.norm(b21 - rdm(rho, 1, 7)))
    print(jnp.linalg.norm(b22 - rdm(rho, 1, 8)))
