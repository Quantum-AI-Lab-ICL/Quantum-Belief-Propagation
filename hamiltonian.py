import jax.numpy as jnp

from const import NUM_PARAMS_SINGLE, NUM_PARAMS_DOUBLE, MATRIX_SIZE_SINGLE, \
    MATRIX_SIZE_DOUBLE
from pauli import Pauli, _pauli_matrix, _pauli_matrix_2d, _parse_pauli_index


class Hamiltonian:
    """
    TODO
    """

    def __init__(self, size: int):
        """
        TODO
        """

        self.size = size
        self._params_single = jnp.zeros((size, NUM_PARAMS_SINGLE),
                                        dtype=jnp.complex64)
        self._params_double = jnp.zeros((size - 1, NUM_PARAMS_DOUBLE),
                                        dtype=jnp.complex64)
        self._reset_partial_hamiltonians()

    def set_param_single(self, index: int, pauli: Pauli, value: int):
        """
        TODO
        """

        pauli_index = pauli.value
        self._params_single = \
            self._params_single.at[index, pauli_index].set(value)

    def set_param_double(self, index: int, pauli_0: Pauli, pauli_1: Pauli,
                         value: int):
        """
        TODO
        """

        pauli_index = _parse_pauli_index(pauli_0, pauli_1)
        self._params_double = \
            self._params_double.at[index, pauli_index].set(value)

    def _reset_partial_hamiltonians(self):
        """
        TODO
        """

        self.hamiltonians = jnp.zeros((self.size - 1, MATRIX_SIZE_DOUBLE,
                                       MATRIX_SIZE_DOUBLE),
                                      dtype=jnp.complex64)

    def compute_partial_hamiltonians(self):
        """
        TODO
        """

        self._reset_partial_hamiltonians()

        ham_single = jnp.zeros((self.size, MATRIX_SIZE_SINGLE,
                                MATRIX_SIZE_SINGLE),
                               dtype=jnp.complex64)

        pauli_matrices = (Pauli.X, Pauli.Y, Pauli.Z)

        for i in range(self.size):
            for pauli in pauli_matrices:
                ham_single = \
                    ham_single.at[i].add(
                        self._params_single[i, pauli.value] *
                        _pauli_matrix(pauli)
                    )

        for i in range(self.size - 1):
            self.hamiltonians = \
                self.hamiltonians.at[i].add(
                    jnp.kron(self._weighted_hamiltonian(ham_single, i),
                             jnp.eye(2)) +
                    jnp.kron(jnp.eye(2),
                             self._weighted_hamiltonian(ham_single, i+1))
                )
            for pauli_0 in pauli_matrices:
                for pauli_1 in pauli_matrices:
                    pauli_index = _parse_pauli_index(pauli_0, pauli_1)
                    self.hamiltonians = \
                        self.hamiltonians.at[i].add(
                            self._params_double[i, pauli_index] *
                            _pauli_matrix_2d(pauli_0, pauli_1)
                        )

    def _weighted_hamiltonian(self, ham_single, index):
        """
        TODO
        """
        if index == 0 or index == self.size - 1:
            return ham_single[index]
        return ham_single[index] / 2

    def get_partial_hamiltonian(self, index):
        """
        TODO
        """

        return self.hamiltonians[index]
