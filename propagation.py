import jax.scipy.linalg as linalg
import jax.numpy as jnp
from jax import random

from const import MATRIX_SIZE_DOUBLE, MATRIX_SIZE_SINGLE
from hamiltonian import Hamiltonian
from utils import _double_to_single_trace, _logmh, _normalise


class BeliefPropagator:
    """
    TODO
    """

    def __init__(self, hamiltonian: Hamiltonian, seed: int):
        """
        TODO
        """

        self.hamiltonian = hamiltonian
        self.key = random.key(seed)
        self.num_beliefs = hamiltonian.size - 1
        self.beliefs = jnp.zeros((self.num_beliefs,
                                  MATRIX_SIZE_DOUBLE,
                                  MATRIX_SIZE_DOUBLE),
                                 dtype=jnp.complex64)
        self.msg_forward = jnp.zeros((self.num_beliefs,
                                      MATRIX_SIZE_SINGLE,
                                      MATRIX_SIZE_SINGLE),
                                     dtype=jnp.complex64)
        self.msg_backward = jnp.zeros((self.num_beliefs,
                                       MATRIX_SIZE_SINGLE,
                                       MATRIX_SIZE_SINGLE),
                                      dtype=jnp.complex64)
        self._initialise()

    def _initialise(self):
        """
        TODO
        """
        for i in range(self.num_beliefs):
            self.beliefs = \
                self.beliefs.at[i].set(jnp.eye(4))
            self.msg_forward = \
                self.msg_forward.at[i].set(jnp.eye(2))
            self.msg_backward = \
                self.msg_backward.at[i].set(jnp.eye(2))

    def step(self):
        """
        TODO
        """

        new_msg_forward = jnp.zeros((self.num_beliefs,
                                     MATRIX_SIZE_SINGLE,
                                     MATRIX_SIZE_SINGLE),
                                    dtype=jnp.complex64)

        new_msg_backward = jnp.zeros((self.num_beliefs,
                                      MATRIX_SIZE_SINGLE,
                                      MATRIX_SIZE_SINGLE),
                                     dtype=jnp.complex64)

        new_msg_forward = \
            new_msg_forward.at[0].set(jnp.eye(2))

        for i in range(1, self.num_beliefs):
            new_msg_forward = \
                new_msg_forward.at[i].set(_normalise(linalg.expm(
                    _logmh(_double_to_single_trace(self.beliefs[i-1], 0)) +
                    _logmh(linalg.inv(self.msg_backward[i-1]))
                )))

        new_msg_backward = \
            new_msg_backward.at[self.num_beliefs - 1].set(jnp.eye(2))

        for i in range(self.num_beliefs - 1):
            new_msg_backward = \
                new_msg_backward.at[i].set(_normalise(linalg.expm(
                    _logmh(_double_to_single_trace(self.beliefs[i+1], 1)) +
                    _logmh(linalg.inv(self.msg_forward[i+1]))
                )))

        self.msg_forward = new_msg_forward
        self.msg_backward = new_msg_backward

        for i in range(self.num_beliefs):
            self.beliefs = \
                self.beliefs.at[i].set(_normalise(linalg.expm(
                    self.hamiltonian.get_partial_hamiltonian(i) +
                    _logmh(jnp.kron(self.msg_forward[i], jnp.eye(2))) +
                    _logmh(jnp.kron(jnp.eye(2), self.msg_backward[i]))
                )))
