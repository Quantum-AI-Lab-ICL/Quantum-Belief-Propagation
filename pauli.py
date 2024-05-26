import jax
import jax.numpy as jnp
import jax.typing
from enum import Enum

from const import MATRIX_SIZE_SINGLE
from utils import tensor_product


class Pauli(Enum):
    """
    TODO
    """
    X = 0
    Y = 1
    Z = 2


def _matrix_X() -> jax.typing.ArrayLike:
    """
    TODO
    """
    return jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)


def _matrix_Y() -> jax.typing.ArrayLike:
    """
    TODO
    """
    return jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)


def _matrix_Z() -> jax.typing.ArrayLike:
    """
    TODO
    """
    return jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)


def _pauli_matrix(pauli: Pauli) -> jax.typing.ArrayLike:
    """
    TODO
    """
    match pauli:
        case Pauli.X:
            return _matrix_X()
        case Pauli.Y:
            return _matrix_Y()
        case Pauli.Z:
            return _matrix_Z()


def _pauli_matrix_2d(pauli_0: Pauli, pauli_1: Pauli) -> jax.typing.ArrayLike:
    """
    TODO
    """
    return jnp.kron(_pauli_matrix(pauli_0), _pauli_matrix(pauli_1))


def _parse_pauli_index(pauli_0: Pauli, pauli_1: Pauli) -> int:
    """
    TODO
    """
    return pauli_0.value * len(Pauli) + pauli_1.value


class CompositePauliMatrix:

    def __init__(self, size: jnp.int32):
        self.size = size
        self.components = jnp.zeros((size, MATRIX_SIZE_SINGLE,
                                     MATRIX_SIZE_SINGLE),
                                    dtype=jnp.complex64)
        for i in range(size):
            self.components = \
                self.components.at[i].set(jnp.eye(MATRIX_SIZE_SINGLE))

    def set_pauli(self, index: jnp.int32, pauli: Pauli):
        self.components = \
            self.components.at[index].set(_pauli_matrix(pauli))

    def get_matrix(self):
        return tensor_product(self.components)
