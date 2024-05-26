import jax.numpy as jnp
from jax.scipy import linalg

from pauli import Pauli
from hamiltonian import Hamiltonian
from propagation import BeliefPropagator
from examples.example_utils import rdm, matrix_3x3, get_single_rho


def main():
    x_coef = -3
    zz_coef = 1
    rsize = 3
    beta = 1

    seed = 0

    H = matrix_3x3(x_coef, zz_coef)
    rho = linalg.expm(-H)
    rho /= jnp.trace(rho)

    rows = [ham_setup(rsize, beta, x_coef, zz_coef) for _ in range(rsize)]
    cols = [ham_setup(rsize, beta, x_coef, zz_coef) for _ in range(rsize)]
    bp_rows = [BeliefPropagator(row, seed) for row in rows]
    bp_cols = [BeliefPropagator(col, seed) for col in cols]

    for _ in range(30):
        for _ in range(rsize):
            for i in range(rsize):
                bp_rows[i].step()

        sh_rows = [get_single_rho(bp_row.beliefs, rsize) for bp_row in bp_rows]

        for i in range(rsize):
            bp_cols[i].beliefs = bp_cols[i].beliefs.at[0].set(
                jnp.kron(sh_rows[0][i], sh_rows[1][i]))
            bp_cols[i].beliefs = bp_cols[i].beliefs.at[1].set(
                jnp.kron(sh_rows[1][i], sh_rows[2][i]))

        for _ in range(rsize):
            for i in range(rsize):
                bp_cols[i].step()

        sh_cols = [get_single_rho(bp_col.beliefs, rsize) for bp_col in bp_cols]

        for i in range(rsize):
            bp_rows[i].beliefs = bp_rows[i].beliefs.at[0].set(
                jnp.kron(sh_cols[0][i], sh_cols[1][i]))
            bp_rows[i].beliefs = bp_rows[i].beliefs.at[1].set(
                jnp.kron(sh_cols[1][i], sh_cols[2][i]))

    print(jnp.linalg.norm(sh_cols[0][0] - rdm(rho, 1, 0)))
    print(jnp.linalg.norm(sh_cols[0][1] - rdm(rho, 1, 5)))
    print(jnp.linalg.norm(sh_cols[2][2] - rdm(rho, 1, 8)))
    print(jnp.linalg.norm(sh_cols[1][1] - rdm(rho, 1, 4)))


def ham_setup(size, beta, x_coef, zz_coef):
    result = Hamiltonian(size, beta)
    for i in range(size):
        result.set_param_single(i, Pauli.X, x_coef)
    for i in range(size - 1):
        result.set_param_double(i, Pauli.Z, Pauli.Z, zz_coef)
    result.compute_partial_hamiltonians()
    return result


if __name__ == "__main__":
    main()
