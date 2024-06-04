import jax.numpy as jnp
from jax.scipy import linalg

from hamiltonian import Hamiltonian
from propagation import BeliefPropagator
from lattice_hamiltonian import LatticeHamiltonian
from lattice_propagation import LatticeBeliefPropagator
from pauli import Pauli

from examples.example_utils import get_single_rho


if __name__ == "__main__":

    x_coef = -2.5
    zz_coef = -1.0
    beta = 0.4
    size = 10

    reg_ham = Hamiltonian(size, beta)
    lat_ham = LatticeHamiltonian(1, size, beta)
    for i in range(size):
        reg_ham.set_param_single(i, Pauli.X, x_coef)
        lat_ham.set_param_single(0, i, Pauli.X, x_coef)
    for i in range(size - 1):
        reg_ham.set_param_double(i, Pauli.Z, Pauli.Z, zz_coef)
        lat_ham.set_param_double_row(0, i, Pauli.Z, Pauli.Z, zz_coef)
    reg_ham.compute_partial_hamiltonians()
    lat_ham.compute_partial_hamiltonians()
    print(reg_ham.hamiltonians)
    print(lat_ham.hams_row)
    print(lat_ham.hams_col)

    reg_bp = BeliefPropagator(reg_ham, 0)
    lat_bp = LatticeBeliefPropagator(lat_ham)

    for i in range(size):
        reg_bp.step()
        lat_bp.step()
        print("Messages", i)
        print(reg_bp.msg_forward)
        print(lat_bp._msg_up_row)

    reg_results = get_single_rho(reg_bp.beliefs, size)
    print("Results")
    print(reg_bp.beliefs)
    print(lat_bp.beliefs_row)
    print(jnp.linalg.norm(reg_results[1] - lat_bp.mean_single_belief(0, 1)))
