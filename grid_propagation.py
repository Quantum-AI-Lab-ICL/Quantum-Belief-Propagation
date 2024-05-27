from jax.scipy import linalg
import jax.numpy as jnp
from jax import random

from const import MATRIX_SIZE_DOUBLE, MATRIX_SIZE_SINGLE
from grid_hamiltonian import GridHamiltonian
from utils import _double_to_single_trace, _logmh, _normalise


NUM_NEIGHBOURS = 6


class GridBeliefPropagator:
    """
    TODO
    """

    def __init__(self, grid_ham: GridHamiltonian):
        """
        TODO
        """

        self._grid_ham = grid_ham
        self.beliefs_row = jnp.zeros((grid_ham.numrows,
                                      grid_ham.numcols - 1,
                                      MATRIX_SIZE_DOUBLE,
                                      MATRIX_SIZE_DOUBLE),
                                     dtype=jnp.complex64)
        self.beliefs_col = jnp.zeros((grid_ham.numcols,
                                      grid_ham.numrows - 1,
                                      MATRIX_SIZE_DOUBLE,
                                      MATRIX_SIZE_DOUBLE),
                                     dtype=jnp.complex64)
        self._msg_from_row = jnp.zeros((grid_ham.numrows,
                                        grid_ham.numcols - 1,
                                        NUM_NEIGHBOURS,
                                        MATRIX_SIZE_SINGLE,
                                        MATRIX_SIZE_SINGLE),
                                       dtype=jnp.complex64)
        self._msg_from_col = jnp.zeros((grid_ham.numcols,
                                        grid_ham.numrows - 1,
                                        NUM_NEIGHBOURS,
                                        MATRIX_SIZE_SINGLE,
                                        MATRIX_SIZE_SINGLE),
                                       dtype=jnp.complex64)

        for r in range(grid_ham.numrows):
            for c in range(grid_ham.numcols - 1):
                self.beliefs_row = \
                    self.beliefs_row.at[r, c].set(jnp.eye(4))
                self._msg_from_row = \
                    self.beliefs_row.at[r, c].set(jnp.eye(2))

        for c in range(grid_ham.numcols):
            for r in range(grid_ham.numrows - 1):
                self.beliefs_col = \
                    self.beliefs_col.at[c, r].set(jnp.eye(4))
                self._msg_from_col = \
                    self.beliefs_col.at[c, r].set(jnp.eye(2))

    def step(self):
        """
        TODO
        """

        new_msg_from_row = jnp.zeros((self._grid_ham.numrows,
                                      self._grid_ham.numcols - 1,
                                      NUM_NEIGHBOURS,
                                      MATRIX_SIZE_SINGLE,
                                      MATRIX_SIZE_SINGLE),
                                     dtype=jnp.complex64)
        new_msg_from_col = jnp.zeros((self._grid_ham.numcols,
                                      self._grid_ham.numrows - 1,
                                      NUM_NEIGHBOURS,
                                      MATRIX_SIZE_SINGLE,
                                      MATRIX_SIZE_SINGLE),
                                     dtype=jnp.complex64)

        for r in range(self._grid_ham.numrows):
            for c in range(self._grid_ham.numcols - 1):
                for i in range(NUM_NEIGHBOURS):
                    flip, pri, sec, trace_ix = \
                        self._find_index(i, r, c)
                    if self._check_index(not flip, pri, sec):
                        rev = self._reverse_index(i)
                        tr_belief = _double_to_single_trace(
                            self.beliefs_row[r, c], trace_ix)
                        if flip:
                            inc_msg = self._msg_from_col[pri, sec, rev]
                        else:
                            inc_msg = self._msg_from_row[pri, sec, rev]
                        inv_inc_msg = linalg.inv(inc_msg)
                        new_msg_from_row = \
                            new_msg_from_row.at[r, c, i].set(
                                _normalise(linalg.expm(
                                    _logmh(tr_belief) +
                                    _logmh(inv_inc_msg)
                                )))

        for c in range(self._grid_ham.numcols):
            for r in range(self._grid_ham.numrows - 1):
                for i in range(NUM_NEIGHBOURS):
                    flip, pri, sec, trace_ix = \
                        self._find_index(i, c, r)
                    if self._check_index(flip, pri, sec):
                        rev = self._reverse_index(i)
                        tr_belief = _double_to_single_trace(
                            self.beliefs_col[c, r], trace_ix)
                        if not flip:
                            inc_msg = self._msg_from_col[pri, sec, rev]
                        else:
                            inc_msg = self._msg_from_row[pri, sec, rev]
                        inv_inc_msg = linalg.inv(inc_msg)
                        new_msg_from_col = \
                            new_msg_from_col.at[c, r, i].set(
                                _normalise(linalg.expm(
                                    _logmh(tr_belief) +
                                    _logmh(inv_inc_msg)
                                )))

        self._msg_from_row = new_msg_from_row
        self._msg_from_col = new_msg_from_col

        for r in range(self._grid_ham.numrows):
            for c in range(self._grid_ham.numcols - 1):
                msg_acc = jnp.zeros((MATRIX_SIZE_DOUBLE,
                                     MATRIX_SIZE_DOUBLE),
                                    dtype=jnp.complex64)
                for i in range(NUM_NEIGHBOURS):
                    flip, pri, sec, trace_ix = \
                        self._find_index(i, r, c)
                    if self._check_index(not flip, pri, sec):
                        rev = self._reverse_index(i)
                        if flip:
                            inc_msg = self._msg_from_col[pri, sec, rev]
                        else:
                            inc_msg = self._msg_from_row[pri, sec, rev]
                        if trace_ix:
                            msg_acc += _logmh(jnp.kron(inc_msg, jnp.eye(2)))
                        else:
                            msg_acc += _logmh(jnp.kron(jnp.eye(2), inc_msg))
                self.beliefs_row = \
                    self.beliefs_row.at[r, c].set(_normalise(linalg.expm(
                        self._grid_ham.get_partial_ham_row(r, c) +
                        msg_acc
                    )))

        for c in range(self._grid_ham.numcols):
            for r in range(self._grid_ham.numrows - 1):
                msg_acc = jnp.zeros((MATRIX_SIZE_DOUBLE,
                                     MATRIX_SIZE_DOUBLE),
                                    dtype=jnp.complex64)
                for i in range(NUM_NEIGHBOURS):
                    flip, pri, sec, trace_ix = \
                        self._find_index(i, c, r)
                    if self._check_index(flip, pri, sec):
                        rev = self._reverse_index(i)
                        if not flip:
                            inc_msg = self._msg_from_col[pri, sec, rev]
                        else:
                            inc_msg = self._msg_from_row[pri, sec, rev]
                        if trace_ix:
                            msg_acc += _logmh(jnp.kron(inc_msg, jnp.eye(2)))
                        else:
                            msg_acc += _logmh(jnp.kron(jnp.eye(2), inc_msg))
                self.beliefs_col = \
                    self.beliefs_col.at[c, r].set(_normalise(linalg.expm(
                        self._grid_ham.get_partial_ham_col(c, r) +
                        msg_acc
                    )))

    def _reverse_index(self, index):
        return index + (NUM_NEIGHBOURS / 2) % NUM_NEIGHBOURS

    def _find_index(self, number, primary, secondary):
        match number:
            case 0:
                return 0, primary, secondary - 1, 1
            case 1:
                return 1, secondary, primary - 1, 1
            case 2:
                return 1, secondary + 1, primary - 1, 0
            case 3:
                return 0, primary, secondary + 1, 0
            case 4:
                return 1, secondary + 1, primary, 0
            case 5:
                return 1, secondary, primary, 0

    def _check_index(self, is_row, primary, secondary):
        if is_row:
            return (primary >= 0 and primary < self.grid_ham.numrows) \
                and (secondary >= 0 and secondary < self.grid_ham.numcols - 1)
        else:
            return (primary >= 0 and primary < self.grid_ham.numcols) \
                and (secondary >= 0 and secondary < self.grid_ham.numrows - 1)
