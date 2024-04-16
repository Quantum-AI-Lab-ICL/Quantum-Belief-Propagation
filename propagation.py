import jax.numpy as jnp
from jax import random

from const import MATRIX_SIZE_SINGLE
from hamiltonian import Hamiltonian


class BeliefPropagator:
    """
    TODO
    """

    def __init__(self, hamiltonian: Hamiltonian, seed: int):
        """
        TODO
        """
        self.hamiltonian = hamiltonian
        self.beliefs = hamiltonian.ham_double
        self.key = random.key(seed)
        self.messages = jnp.zeros(((hamiltonian.size - 1) * 2,
                                   MATRIX_SIZE_SINGLE, MATRIX_SIZE_SINGLE),
                                  dtype=jnp.complex64)

    def step(self):
        """
        TODO
        """
