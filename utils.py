import jax.numpy as jnp
import jax.typing
from jax import Array


def _bra_0():
    """
    TODO
    """

    return jnp.array([[1, 0]], dtype=jnp.complex64)


def _bra_1():
    """
    TODO
    """

    return jnp.array([[0, 1]], dtype=jnp.complex64)


def _ket_0():
    """
    TODO
    """

    return jnp.array([[1], [0]], dtype=jnp.complex64)


def _ket_1():
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
