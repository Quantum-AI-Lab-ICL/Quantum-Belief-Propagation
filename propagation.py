"""
BeliefPropagator class for running the 1D algorithm.
"""
import jax.numpy as jnp
import jax.scipy.linalg as linalg

from const import MATRIX_SIZE_DOUBLE, MATRIX_SIZE_SINGLE
from hamiltonian import Hamiltonian
from utils import _double_to_single_trace, _logmh, _normalise


class BeliefPropagator:
    """
    Class for performing the quantum belief propagation algorithm in 1D.

    Attributes
    ----------
    hamiltonian: Hamiltonian
        the Hamiltonian of the underlying quantum system
    beliefs: jax.Array
        the pair-wise beliefs for the reduced density matrices

    Methods
    -------
    step
        run one step of the algorithm as specified, including the computation
        of the messages and the new beliefs
    """

    def __init__(self, hamiltonian: Hamiltonian):
        """
        Initialise the BeliefPropagator class.

        Parameters
        ----------
        hamiltonian: Hamiltonian
            the Hamiltonian of the underlying quantum system
        """

        self.hamiltonian = hamiltonian
        self._num_beliefs = hamiltonian.size - 1
        self.beliefs = jnp.zeros((self._num_beliefs,
                                  MATRIX_SIZE_DOUBLE,
                                  MATRIX_SIZE_DOUBLE),
                                 dtype=jnp.complex64)
        self._msg_forward = jnp.zeros((self._num_beliefs,
                                       MATRIX_SIZE_SINGLE,
                                       MATRIX_SIZE_SINGLE),
                                      dtype=jnp.complex64)
        self._msg_backward = jnp.zeros((self._num_beliefs,
                                        MATRIX_SIZE_SINGLE,
                                        MATRIX_SIZE_SINGLE),
                                       dtype=jnp.complex64)
        self._initialise()

    def _initialise(self):
        """
        Initialise the beliefs and the messages.
        """
        for i in range(self._num_beliefs):
            self.beliefs = \
                self.beliefs.at[i].set(jnp.eye(4))
            self._msg_forward = \
                self._msg_forward.at[i].set(jnp.eye(2))
            self._msg_backward = \
                self._msg_backward.at[i].set(jnp.eye(2))

    def step(self):
        """
        Run one step of the algorithm as specified. This includes the
        computation of the messages and the new beliefs.
        """

        new_msg_forward = jnp.zeros((self._num_beliefs,
                                     MATRIX_SIZE_SINGLE,
                                     MATRIX_SIZE_SINGLE),
                                    dtype=jnp.complex64)

        new_msg_backward = jnp.zeros((self._num_beliefs,
                                      MATRIX_SIZE_SINGLE,
                                      MATRIX_SIZE_SINGLE),
                                     dtype=jnp.complex64)

        new_msg_forward = \
            new_msg_forward.at[0].set(jnp.eye(2) / 2)

        for i in range(1, self._num_beliefs):
            new_msg_forward = \
                new_msg_forward.at[i].set(_normalise(linalg.expm(
                    _logmh(_double_to_single_trace(self.beliefs[i-1], 0)) +
                    _logmh(linalg.inv(self._msg_backward[i-1]))
                )))

        new_msg_backward = \
            new_msg_backward.at[self._num_beliefs - 1].set(jnp.eye(2) / 2)

        for i in range(self._num_beliefs - 1):
            new_msg_backward = \
                new_msg_backward.at[i].set(_normalise(linalg.expm(
                    _logmh(_double_to_single_trace(self.beliefs[i+1], 1)) +
                    _logmh(linalg.inv(self._msg_forward[i+1]))
                )))

        self._msg_forward = new_msg_forward
        self._msg_backward = new_msg_backward

        for i in range(self._num_beliefs):
            self.beliefs = \
                self.beliefs.at[i].set(_normalise(linalg.expm(
                    self.hamiltonian.get_partial_hamiltonian(i) +
                    _logmh(jnp.kron(self._msg_forward[i], jnp.eye(2))) +
                    _logmh(jnp.kron(jnp.eye(2), self._msg_backward[i]))
                )))
