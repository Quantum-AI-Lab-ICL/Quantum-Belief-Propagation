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

    exact = jnp.zeros((size, 2, 2), dtype=jnp.complex64)
    for i in range(size):
        exact = exact.at[i].set(rdm(rho, 1, i))

    ham = Hamiltonian(size, beta=1)
    for i in range(size):
        ham.set_param_single(i, Pauli.X, x_coef)
    for i in range(size - 1):
        ham.set_param_double(i, Pauli.Z, Pauli.Z, zz_coef)
    ham.compute_partial_hamiltonians()
    bp = BeliefPropagator(ham, 0)
    for _ in range(3):
        bp.step()
    sr = get_single_rho(bp.beliefs, size)

    error = 0
    for i in range(size):
        error += jnp.linalg.norm(exact[i] - sr[i])
    print(error / size)

    bp_s2 = BeliefPropagator(ham, 0)
    bp_s2.beliefs = bp_s2.beliefs.at[0].set(jnp.kron(sr[0], sr[5]))
    bp_s2.beliefs = bp_s2.beliefs.at[1].set(jnp.kron(sr[5], sr[6]))
    bp_s2.beliefs = bp_s2.beliefs.at[0].set(jnp.kron(sr[6], sr[7]))
    bp_s2.beliefs = bp_s2.beliefs.at[0].set(jnp.kron(sr[7], sr[4]))
    bp_s2.beliefs = bp_s2.beliefs.at[0].set(jnp.kron(sr[4], sr[1]))
    bp_s2.beliefs = bp_s2.beliefs.at[0].set(jnp.kron(sr[1], sr[2]))
    bp_s2.beliefs = bp_s2.beliefs.at[0].set(jnp.kron(sr[2], sr[3]))
    bp_s2.beliefs = bp_s2.beliefs.at[0].set(jnp.kron(sr[3], sr[8]))
    for i in range(size):
        print(f"step {i}")
        bp_s2.step()

        error = 0
        srs2 = get_single_rho(bp_s2.beliefs, size)
        for i in range(size):
            error += jnp.linalg.norm(exact[i] - srs2[i])
        print(error / size)


if __name__ == "__main__":
    main()
