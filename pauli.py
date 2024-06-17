"""
Classes and functions relating to the Pauli matrix.
"""
import jax
import jax.numpy as jnp
import jax.typing
from enum import Enum

from const import MATRIX_SIZE_SINGLE
from utils import _tensor_product


class Pauli(Enum):
    """
    Class that represents the types of Pauli matrices.
    """
    X = 0
    Y = 1
    Z = 2


def _matrix_X() -> jax.typing.ArrayLike:
    """The Pauli X matrix."""
    return jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)


def _matrix_Y() -> jax.typing.ArrayLike:
    """The Pauli Y matrix."""
    return jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)


def _matrix_Z() -> jax.typing.ArrayLike:
    """The Pauli Z matrix."""
    return jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)


def _pauli_matrix(pauli: Pauli) -> jax.typing.ArrayLike:
    """Returns the Pauli matrix corresponding to the Pauli value."""
    match pauli:
        case Pauli.X:
            return _matrix_X()
        case Pauli.Y:
            return _matrix_Y()
        case Pauli.Z:
            return _matrix_Z()


def _pauli_matrix_2d(pauli_0: Pauli, pauli_1: Pauli) -> jax.typing.ArrayLike:
    """Returns the 2D Pauli matrix corresponding to the Pauli values."""
    return jnp.kron(_pauli_matrix(pauli_0), _pauli_matrix(pauli_1))


def _parse_pauli_index(pauli_0: Pauli, pauli_1: Pauli) -> jnp.int32:
    """Parse the Pauli values into an integer index."""
    return pauli_0.value * len(Pauli) + pauli_1.value


class CompositePauliMatrix:
    """
    Class that represents compositions of Pauli matrices.

    Attributes
    ----------
    size: jnp.int32
        size of the system, i.e. number of particles
    components: jax.Array
        component 2D matrices that make up the matrix

    Methods
    -------
    set_pauli
        set the component at the index to the type of pauli matrix
    get_matrix
        get the matrix corresponding to the components
    """

    def __init__(self, size: jnp.int32):
        """
        Initialise the CompositePauliMatrix class, setting all components to
        the identity matrix.

        Parameters
        ----------
        size: jnp.int32
            size of the system, i.e. number of particles
        """
        self.size = size
        self.components = jnp.zeros((size, MATRIX_SIZE_SINGLE,
                                     MATRIX_SIZE_SINGLE),
                                    dtype=jnp.complex64)
        for i in range(size):
            self.components = \
                self.components.at[i].set(jnp.eye(MATRIX_SIZE_SINGLE))

    def set_pauli(self, index: jnp.int32, pauli: Pauli):
        """
        Set the component at the index to the type of pauli matrix.

        Parameters
        ----------
        index: jnp.int32
            index of the particle at which to set the Pauli matrix
        pauli: Pauli
            the type of Pauli matrix
        """
        self.components = \
            self.components.at[index].set(_pauli_matrix(pauli))

    def get_matrix(self) -> jax.Array:
        """
        Get the matrix corresponding to the components.

        Returns
        -------
        matrix:
            the resultin matrix
        """
        return _tensor_product(self.components)
        return tensor_product(self.components)
