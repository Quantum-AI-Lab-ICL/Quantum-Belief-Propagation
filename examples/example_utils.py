import jax.numpy as jnp
import jax.typing
from const import MATRIX_SIZE_SINGLE
from hamiltonian import Hamiltonian


def hamiltonian_matrix(ham: Hamiltonian) -> jax.typing.ArrayLike:
    """
    TODO
    """

    total_size = MATRIX_SIZE_SINGLE ** ham.size
    total = jnp.zeros((total_size, total_size), dtype=jnp.complex64)

    for i in range(ham.size - 1):
        total = total + jnp.kron(
            jnp.kron(jnp.eye(MATRIX_SIZE_SINGLE ** i),
                     ham.get_partial_hamiltonian(i)),
            jnp.eye(MATRIX_SIZE_SINGLE ** (ham.size - i - 2))
        )

    return total


def bra(n, d):
    """
    Return the 2D array representation of the bra vector with d digits
    representing the numeric value n.
    """
    # Check that the ket number is non-negative
    assert n >= 0
    # Check that the number of digits is positive
    assert d > 0

    size = jnp.power(2, d).astype(jnp.int32)

    # Check that the binary representation is contained within the
    # number of digits
    assert size > n

    psi = jnp.zeros((1, size))
    psi = psi.at[0, n].set(1)

    return psi


def ket(n, d):
    """
    Return the 2D array representation of the ket vector with d digits
    representing the numeric value n.
    """
    # Check that the ket number is non-negative
    assert n >= 0
    # Check that the number of digits is positive
    assert d > 0

    size = jnp.power(2, d).astype(jnp.int32)

    # Check that the binary representation is contained within the
    # number of digits
    assert size > n

    psi = jnp.zeros((size, 1))
    psi = psi.at[n, 0].set(1)

    return psi


def rdm(rho, partial_dim, pos):
    """
    TODO
    """

    result_size = MATRIX_SIZE_SINGLE * partial_dim
    result = jnp.zeros((result_size, result_size))

    total_dim = jnp.log2(rho.shape[0]).astype(jnp.int32)
    ld = pos
    rd = total_dim - partial_dim - pos

    for ln in range(int(MATRIX_SIZE_SINGLE ** ld)):
        for rn in range(int(MATRIX_SIZE_SINGLE ** rd)):
            l_bra = bra(ln, ld) if ld != 0 else 1
            r_bra = bra(rn, rd) if rd != 0 else 1
            bra_side = jnp.kron(jnp.kron(l_bra, jnp.eye(result_size)), r_bra)
            l_ket = ket(ln, ld) if ld != 0 else 1
            r_ket = ket(rn, rd) if rd != 0 else 1
            ket_side = jnp.kron(jnp.kron(l_ket, jnp.eye(result_size)), r_ket)
            result += bra_side @ rho @ ket_side

    return result