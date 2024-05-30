import jax.numpy as jnp

from const import NUM_PARAMS_SINGLE, NUM_PARAMS_DOUBLE, MATRIX_SIZE_SINGLE, \
    MATRIX_SIZE_DOUBLE
from pauli import Pauli, _pauli_matrix, _pauli_matrix_2d, _parse_pauli_index


class LatticeHamiltonian:
    """
    TODO
    """

    def __init__(self, numrows: jnp.int32, numcols: jnp.int32,
                 beta: jnp.complex64):
        """
        TODO
        """

        self.numrows = numrows
        self.numcols = numcols
        self.beta = beta
        self._params_single = jnp.zeros((numrows,
                                         numcols,
                                         NUM_PARAMS_SINGLE),
                                        dtype=jnp.complex64)
        self._params_double_row = jnp.zeros((numrows,
                                             numcols - 1,
                                             NUM_PARAMS_DOUBLE),
                                            dtype=jnp.complex64)
        self._params_double_col = jnp.zeros((numcols,
                                             numrows - 1,
                                             NUM_PARAMS_DOUBLE),
                                            dtype=jnp.complex64)
        self._reset_partial_hamiltonians()

    def set_param_single(self, rowindex: jnp.int32, colindex: jnp.int32,
                         pauli: Pauli, value: jnp.complex64):
        """
        TODO
        """

        pauli_index = pauli.value
        self._params_single = \
            self._params_single.at[rowindex, colindex, pauli_index] \
            .set(-self.beta * value)

    def set_param_double_row(self, rowindex: jnp.int32, edgeindex: jnp.int32,
                             pauli_0: Pauli, pauli_1: Pauli,
                             value: jnp.complex64):
        """
        TODO
        """

        pauli_index = _parse_pauli_index(pauli_0, pauli_1)
        self._params_double_row = \
            self._params_double_row.at[rowindex, edgeindex, pauli_index] \
            .set(-self.beta * value)

    def set_param_double_col(self, colindex: jnp.int32, edgeindex: jnp.int32,
                             pauli_0: Pauli, pauli_1: Pauli,
                             value: jnp.complex64):
        """
        TODO
        """

        pauli_index = _parse_pauli_index(pauli_0, pauli_1)
        self._params_double_col = \
            self._params_double_col.at[colindex, edgeindex, pauli_index] \
            .set(-self.beta * value)

    def _reset_partial_hamiltonians(self):
        """
        TODO
        """

        self.hams_row = jnp.zeros((self.numrows,
                                   self.numcols - 1,
                                   MATRIX_SIZE_DOUBLE,
                                   MATRIX_SIZE_DOUBLE),
                                  dtype=jnp.complex64)
        self.hams_col = jnp.zeros((self.numcols,
                                   self.numrows - 1,
                                   MATRIX_SIZE_DOUBLE,
                                   MATRIX_SIZE_DOUBLE),
                                  dtype=jnp.complex64)

    def compute_partial_hamiltonians(self):
        """
        TODO
        """

        self._reset_partial_hamiltonians()

        hams_single = jnp.zeros((self.numrows,
                                 self.numcols,
                                 MATRIX_SIZE_SINGLE,
                                 MATRIX_SIZE_SINGLE),
                                dtype=jnp.complex64)

        pauli_matrices = (Pauli.X, Pauli.Y, Pauli.Z)

        for r in range(self.numrows):
            for c in range(self.numcols):
                for pauli in pauli_matrices:
                    hams_single = \
                        hams_single.at[r, c].add(
                            self._params_single[r, c, pauli.value] *
                            _pauli_matrix(pauli)
                        )

        # Rows
        for r in range(self.numrows):
            for c in range(self.numcols - 1):
                self.hams_row = \
                    self.hams_row.at[r, c].add(
                        jnp.kron(self._weighted_ham(hams_single, r, c),
                                 jnp.eye(2)) +
                        jnp.kron(jnp.eye(2),
                                 self._weighted_ham(hams_single, r, c+1))
                    )
                for pauli_0 in pauli_matrices:
                    for pauli_1 in pauli_matrices:
                        pauli_index = _parse_pauli_index(pauli_0, pauli_1)
                        self.hams_row = \
                            self.hams_row.at[r, c].add(
                                self._params_double_row[r, c, pauli_index] *
                                _pauli_matrix_2d(pauli_0, pauli_1)
                            )

        # Cols
        for c in range(self.numcols):
            for r in range(self.numrows - 1):
                self.hams_col = \
                    self.hams_col.at[c, r].add(
                        jnp.kron(self._weighted_ham(hams_single, r, c),
                                 jnp.eye(2)) +
                        jnp.kron(jnp.eye(2),
                                 self._weighted_ham(hams_single, r+1, c))
                    )
                for pauli_0 in pauli_matrices:
                    for pauli_1 in pauli_matrices:
                        pauli_index = _parse_pauli_index(pauli_0, pauli_1)
                        self.hams_col = \
                            self.hams_col.at[c, r].add(
                                self._params_double_row[c, r, pauli_index] *
                                _pauli_matrix_2d(pauli_0, pauli_1)
                            )

    def _weighted_ham(self, hams_single, rowindex, colindex):
        """
        TODO
        """

        num_edges = 4
        if rowindex == 0 or rowindex == self.numrows - 1:
            num_edges -= 1
        if colindex == 0 or colindex == self.numcols - 1:
            num_edges -= 1
        return hams_single[rowindex, colindex] / num_edges

    def get_partial_ham_row(self, rowindex, edgeindex):
        """
        TODO
        """

        return self.hams_row[rowindex, edgeindex]

    def get_partial_ham_col(self, colindex, edgeindex):
        """
        TODO
        """

        return self.hams_col[colindex, edgeindex]
