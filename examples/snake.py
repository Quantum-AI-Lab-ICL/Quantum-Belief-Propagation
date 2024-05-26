import jax.numpy as jnp
from jax.scipy import linalg
import sys

from pauli import Pauli
from examples.example_utils import rdm, get_single_rho, matrix_3x3
from hamiltonian import Hamiltonian
from propagation import BeliefPropagator


def main():
    x_coef = -3
    zz_coef = 1
    size = 9
    jnp.set_printoptions(linewidth=10000, threshold=sys.maxsize)
    H = matrix_3x3(x_coef, zz_coef)
    rho = linalg.expm(-H)
    rho /= jnp.trace(rho)

    ham = Hamiltonian(size, beta=1)
    for i in range(size):
        ham.set_param_single(i, Pauli.X, x_coef)
    for i in range(size - 1):
        ham.set_param_double(i, Pauli.Z, Pauli.Z, zz_coef)
    ham.compute_partial_hamiltonians()
    bp = BeliefPropagator(ham, 0)
    for _ in range(size):
        bp.step()
    single_rho = get_single_rho(bp.beliefs, size)
    error = 0
    for i in range(size):
        error += jnp.linalg.norm(rdm(rho, 1, i) - single_rho[i])
    print(error / size)


if __name__ == "__main__":
    main()
