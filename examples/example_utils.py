import jax.numpy as jnp
import jax.typing
from const import MATRIX_SIZE_SINGLE
from hamiltonian import Hamiltonian
from utils import _double_to_single_trace
from pauli import CompositePauliMatrix, Pauli, _matrix_X, _pauli_matrix_2d


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


def get_single_rho(beliefs, size):
    results = jnp.zeros((size, MATRIX_SIZE_SINGLE, MATRIX_SIZE_SINGLE),
                        dtype=jnp.complex64)
    results = results.at[0].set(
        _double_to_single_trace(beliefs[0], 1))
    results = results.at[size-1].set(
        _double_to_single_trace(beliefs[size-2], 0))
    for i in range(1, beliefs.shape[0]):
        results = results.at[i].set(
            (_double_to_single_trace(beliefs[i-1], 0) +
             _double_to_single_trace(beliefs[i], 1)) / 2
        )
    return results


def trans_mag(single_rho):
    """
    TODO
    """

    result = 0
    for i in range(single_rho.shape[0]):
        result += jnp.trace(single_rho[i] @ _matrix_X())
    return result / single_rho.shape[0]


def correlation(double_rho):
    """
    TODO
    """

    result = 0
    for i in range(double_rho.shape[0]):
        result += jnp.trace(double_rho[i] @ _pauli_matrix_2d(Pauli.X, Pauli.X))
    return result / double_rho.shape[0]


def correlation_spin_2d(double_rho):
    """
    TODO
    """

    result = 0
    for i in range(double_rho.shape[0]):
        for j in range(double_rho.shape[1]):
            result += jnp.trace(
                double_rho[i, j] @ _pauli_matrix_2d(Pauli.Z, Pauli.Z))
    return result / (double_rho.shape[0] * double_rho.shape[1])


def matrix_3x3(x_coef: jnp.int32, zz_coef: jnp.int32):
    """
    TODO
    """
    size = 9
    matrix = jnp.zeros((2**size, 2**size), dtype=jnp.complex64)
    for i in range(size):
        matrix += x_coef * x_component(size, i).get_matrix()
    edges = {
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (0, 5), (1, 4), (3, 8), (4, 7)
    }
    for ixs in edges:
        matrix += zz_coef * zz_component(size, ixs[0], ixs[1]).get_matrix()
    return matrix


def matrix_4x4(x_coef: jnp.int32, zz_coef: jnp.int32):
    """
    TODO
    """
    size = 16
    matrix = jnp.zeros((2**size, 2**size), dtype=jnp.complex64)
    for i in range(size):
        matrix += x_coef * x_component(size, i).get_matrix()
    edges = {
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
        (0, 7), (1, 6), (2, 5), (6, 9), (5, 10), (4, 11), (10, 13), (9, 14),
        (8, 15)
    }
    for ixs in edges:
        matrix += zz_coef * zz_component(size, ixs[0], ixs[1]).get_matrix()
    return matrix


def x_component(size: jnp.int32, ix: jnp.int32):
    """
    TODO
    """
    cpm = CompositePauliMatrix(size)
    cpm.set_pauli(ix, Pauli.X)
    return cpm


def zz_component(size: jnp.int32, ix1: jnp.int32, ix2: jnp.int32):
    """
    TODO
    """
    cpm = CompositePauliMatrix(size)
    cpm.set_pauli(ix1, Pauli.Z)
    cpm.set_pauli(ix2, Pauli.Z)
    return cpm
