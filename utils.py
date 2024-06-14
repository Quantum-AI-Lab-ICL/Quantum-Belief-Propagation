import jax.numpy as jnp
import jax.typing
from jax.scipy import linalg
from jax import Array
from jax import random


def _bra_0() -> Array:
    """
    TODO
    """

    return jnp.array([[1, 0]], dtype=jnp.complex64)


def _bra_1() -> Array:
    """
    TODO
    """

    return jnp.array([[0, 1]], dtype=jnp.complex64)


def _ket_0() -> Array:
    """
    TODO
    """

    return jnp.array([[1], [0]], dtype=jnp.complex64)


def _ket_1() -> Array:
    """
    TODO
    """

    return jnp.array([[0], [1]], dtype=jnp.complex64)


def _double_to_single_trace(rho_double: jax.typing.ArrayLike,
                            trace_index: int) -> Array:
    """
    TODO
    """

    if trace_index == 0:
        return jnp.kron(_bra_0(), jnp.eye(2)) @ rho_double @ \
            jnp.kron(_ket_0(), jnp.eye(2)) + \
            jnp.kron(_bra_1(), jnp.eye(2)) @ rho_double @ \
            jnp.kron(_ket_1(), jnp.eye(2))
    else:
        return jnp.kron(jnp.eye(2), _bra_0()) @ rho_double @ \
            jnp.kron(jnp.eye(2), _ket_0()) + \
            jnp.kron(jnp.eye(2), _bra_1()) @ rho_double @ \
            jnp.kron(jnp.eye(2), _ket_1())


def _normalise(rho: jax.typing.ArrayLike) -> Array:
    """
    Normalise operator so that its trace is 1
    """

    rho /= jnp.trace(rho)
    return rho


def tensor_product(operators: jax.typing.ArrayLike) -> Array:
    """
    Calculate the tensor product of the operators.
    """

    accumulator = jnp.array(1.0, dtype=jnp.complex64)
    [accumulator := jnp.kron(accumulator, x) for x in operators]
    return accumulator


def _logmh(rho: jax.typing.ArrayLike) -> Array:
    """
    TODO
    """

    eigvals, eigvecs = linalg.eigh(rho)
    log_eigvals = jnp.log(eigvals)
    result = eigvecs @ jnp.diag(log_eigvals) \
        @ jnp.conjugate(jnp.transpose(eigvecs))
    return result


def _random_normalised_hermitian(
        size: jnp.int32, key: jax.typing.ArrayLike) -> (Array, Array):
    factor = 3
    real_key, imag_key = random.split(key)
    real = random.uniform(real_key, (size, size))
    imag = random.uniform(imag_key, (size, size))
    matrix = real + 1.0j * imag
    hermitian = matrix + matrix.transpose().conjugate()
    result = _normalise(jnp.eye(size) + factor * hermitian)
    new_key, _ = random.split(real_key)
    return result, new_key
