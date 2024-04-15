import jax.numpy as jnp
from pauli import Pauli, _pauli_matrix, _pauli_matrix_2d, _parse_pauli_index


NUM_PARAMS_SINGLE = 3
NUM_PARAMS_DOUBLE = 9
MATRIX_SIZE_SINGLE = 2
MATRIX_SIZE_DOUBLE = 4


class Hamiltonian:
    """
    TODO
    """

    def __init__(self, size: int):
        """
        TODO
        """

        self.size = size
        self.params_single = jnp.zeros((size, NUM_PARAMS_SINGLE),
                                       dtype=jnp.complex64)
        self.params_double = jnp.zeros((size - 1, NUM_PARAMS_DOUBLE),
                                       dtype=jnp.complex64)
        self._clear_partial_hamiltonians()

    def set_param_single(self, index: int, pauli: Pauli, value: int):
        """
        TODO
        """

        pauli_index = pauli.value
        self.params_single = \
            self.params_single.at[index, pauli_index].set(value)

    def set_param_double(self, index: int, pauli_0: Pauli, pauli_1: Pauli,
                         value: int):
        """
        TODO
        """

        pauli_index = _parse_pauli_index(pauli_0, pauli_1)
        self.params_double = \
            self.params_double.at[index, pauli_index].set(value)

    def _clear_partial_hamiltonians(self):
        """
        TODO
        """

        self.ham_single = jnp.zeros((self.size, MATRIX_SIZE_SINGLE,
                                     MATRIX_SIZE_SINGLE),
                                    dtype=jnp.complex64)
        self.ham_double = jnp.zeros((self.size - 1, MATRIX_SIZE_DOUBLE,
                                     MATRIX_SIZE_DOUBLE),
                                    dtype=jnp.complex64)

    def compute_partial_hamiltonians(self):
        """
        TODO
        """

        self._clear_partial_hamiltonians()

        pauli_matrices = (Pauli.X, Pauli.Y, Pauli.Z)

        for i in range(self.size):
            for pauli in pauli_matrices:
                self.ham_single = \
                    self.ham_single.at[i].add(
                        self.params_single[i, pauli.value] *
                        _pauli_matrix(pauli)
                    )

        for i in range(self.size - 1):
            self.ham_double = \
                self.ham_double.at[i].add(
                    jnp.kron(self.ham_single[i], jnp.eye(2)) +
                    jnp.kron(jnp.eye(2), self.ham_single[i+1])
                )
            for pauli_0 in pauli_matrices:
                for pauli_1 in pauli_matrices:
                    pauli_index = _parse_pauli_index(pauli_0, pauli_1)
                    self.ham_double = \
                        self.ham_double.at[i].add(
                            self.params_double[i, pauli_index] *
                            _pauli_matrix_2d(pauli_0, pauli_1)
                        )
