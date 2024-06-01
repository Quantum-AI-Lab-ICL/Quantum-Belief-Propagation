from jax.scipy import linalg
import jax.numpy as jnp
from jax import random

from const import MATRIX_SIZE_DOUBLE, MATRIX_SIZE_SINGLE
from lattice_hamiltonian import LatticeHamiltonian
from utils import _double_to_single_trace, _logmh, _normalise


NUM_NEIGHBOURS = 6


class LatticeBeliefPropagator:
    """
    TODO
    """

    def __init__(self, lat_ham: LatticeHamiltonian):
        """
        TODO
        """

        self._lat_ham = lat_ham
        self.beliefs_row = jnp.zeros((lat_ham.numrows,
                                      lat_ham.numcols - 1,
                                      MATRIX_SIZE_DOUBLE,
                                      MATRIX_SIZE_DOUBLE),
                                     dtype=jnp.complex64)
        self.beliefs_col = jnp.zeros((lat_ham.numcols,
                                      lat_ham.numrows - 1,
                                      MATRIX_SIZE_DOUBLE,
                                      MATRIX_SIZE_DOUBLE),
                                     dtype=jnp.complex64)
        self._msg_up_row = jnp.zeros((lat_ham.numrows,
                                      lat_ham.numcols - 1,
                                      MATRIX_SIZE_SINGLE,
                                      MATRIX_SIZE_SINGLE),
                                     dtype=jnp.complex64)
        self._msg_down_row = jnp.zeros((lat_ham.numrows,
                                        lat_ham.numcols - 1,
                                        MATRIX_SIZE_SINGLE,
                                        MATRIX_SIZE_SINGLE),
                                       dtype=jnp.complex64)
        self._msg_up_col = jnp.zeros((lat_ham.numcols,
                                      lat_ham.numrows - 1,
                                      MATRIX_SIZE_SINGLE,
                                      MATRIX_SIZE_SINGLE),
                                     dtype=jnp.complex64)
        self._msg_down_col = jnp.zeros((lat_ham.numcols,
                                        lat_ham.numrows - 1,
                                        MATRIX_SIZE_SINGLE,
                                        MATRIX_SIZE_SINGLE),
                                       dtype=jnp.complex64)

        for r in range(lat_ham.numrows):
            for c in range(lat_ham.numcols - 1):
                self.beliefs_row = \
                    self.beliefs_row.at[r, c].set(jnp.eye(4))
                self._msg_up_row = \
                    self._msg_up_row.at[r, c].set(jnp.eye(2))
                self._msg_down_row = \
                    self._msg_down_row.at[r, c].set(jnp.eye(2))

        for c in range(lat_ham.numcols):
            for r in range(lat_ham.numrows - 1):
                self.beliefs_col = \
                    self.beliefs_col.at[c, r].set(jnp.eye(4))
                self._msg_up_col = \
                    self._msg_up_col.at[c, r].set(jnp.eye(2))
                self._msg_down_col = \
                    self._msg_down_col.at[c, r].set(jnp.eye(2))

    def step(self):
        """
        TODO
        """

        new_msg_up_row = jnp.zeros((self._lat_ham.numrows,
                                    self._lat_ham.numcols - 1,
                                    MATRIX_SIZE_SINGLE,
                                    MATRIX_SIZE_SINGLE),
                                   dtype=jnp.complex64)
        new_msg_down_row = jnp.zeros((self._lat_ham.numrows,
                                      self._lat_ham.numcols - 1,
                                      MATRIX_SIZE_SINGLE,
                                      MATRIX_SIZE_SINGLE),
                                     dtype=jnp.complex64)
        new_msg_up_col = jnp.zeros((self._lat_ham.numcols,
                                    self._lat_ham.numrows - 1,
                                    MATRIX_SIZE_SINGLE,
                                    MATRIX_SIZE_SINGLE),
                                   dtype=jnp.complex64)
        new_msg_down_col = jnp.zeros((self._lat_ham.numcols,
                                      self._lat_ham.numrows - 1,
                                      MATRIX_SIZE_SINGLE,
                                      MATRIX_SIZE_SINGLE),
                                     dtype=jnp.complex64)

        def checked_logmh_inv(msgs, ix0, ix1):
            if 0 <= ix0 and ix0 < msgs.shape[0] \
                    and 0 <= ix1 and ix1 < msgs.shape[1]:
                return _logmh(linalg.inv(msgs[ix0, ix1]))
            return jnp.zeros((2, 2), dtype=jnp.complex64)

        def checked_trace(beliefs, ix0, ix1, trace_id):
            if 0 <= ix0 and ix0 < beliefs.shape[0] \
                    and 0 <= ix1 and ix1 < beliefs.shape[1]:
                return _double_to_single_trace(beliefs[ix0, ix1], trace_id)
            return None

        def mean_log_trace(list_trace):
            sum_trace = jnp.zeros((2, 2), dtype=jnp.complex64)
            count = 0
            for t in list_trace:
                if t is not None:
                    sum_trace += _logmh(t)
                    count += 1
            return sum_trace

        for r in range(self._lat_ham.numrows):
            for c in range(self._lat_ham.numcols - 1):
                up_mean_trace = mean_log_trace([
                    checked_trace(self.beliefs_row, r, c-1, 0),
                    checked_trace(self.beliefs_col, c, r-1, 0),
                    checked_trace(self.beliefs_col, c, r, 1),
                ])
                new_msg_up_row = \
                    new_msg_up_row.at[r, c].set(_normalise(linalg.expm(
                        up_mean_trace +
                        checked_logmh_inv(self._msg_down_row, r, c-1) +
                        checked_logmh_inv(self._msg_down_col, c, r-1) +
                        checked_logmh_inv(self._msg_up_col, c, r)
                    )))

                down_mean_trace = mean_log_trace([
                    checked_trace(self.beliefs_row, r, c+1, 1),
                    checked_trace(self.beliefs_col, c+1, r, 1),
                    checked_trace(self.beliefs_col, c+1, r-1, 0),
                ])
                new_msg_down_row = \
                    new_msg_down_row.at[r, c].set(_normalise(linalg.expm(
                        down_mean_trace +
                        checked_logmh_inv(self._msg_up_row, r, c+1) +
                        checked_logmh_inv(self._msg_up_col, c+1, r) +
                        checked_logmh_inv(self._msg_down_col, c+1, r-1)
                    )))

        for c in range(self._lat_ham.numcols):
            for r in range(self._lat_ham.numrows - 1):

                up_mean_trace = mean_log_trace([
                    checked_trace(self.beliefs_col, c, r-1, 0),
                    checked_trace(self.beliefs_row, r, c-1, 0),
                    checked_trace(self.beliefs_row, r, c, 1),
                ])
                new_msg_up_col = \
                    new_msg_up_col.at[c, r].set(_normalise(linalg.expm(
                        up_mean_trace +
                        checked_logmh_inv(self._msg_down_col, c, r-1) +
                        checked_logmh_inv(self._msg_down_row, r, c-1) +
                        checked_logmh_inv(self._msg_up_row, r, c)
                    )))

                down_mean_trace = mean_log_trace([
                    checked_trace(self.beliefs_col, c, r+1, 1),
                    checked_trace(self.beliefs_row, r+1, c, 1),
                    checked_trace(self.beliefs_row, r+1, c-1, 0),
                ])
                new_msg_down_col = \
                    new_msg_down_col.at[c, r].set(_normalise(linalg.expm(
                        down_mean_trace +
                        checked_logmh_inv(self._msg_up_col, c, r+1) +
                        checked_logmh_inv(self._msg_up_row, r+1, c) +
                        checked_logmh_inv(self._msg_down_row, r+1, c-1)
                    )))

        self._msg_up_row = new_msg_up_row
        self._msg_down_row = new_msg_down_row
        self._msg_up_col = new_msg_up_col
        self._msg_down_col = new_msg_down_col

        for r in range(self._lat_ham.numrows):
            for c in range(self._lat_ham.numcols - 1):
                self.beliefs_row = \
                    self.beliefs_row.at[r, c].set(_normalise(linalg.expm(
                        self._lat_ham.get_partial_ham_row(r, c) +
                        _logmh(jnp.kron(self._msg_up_row[r, c], jnp.eye(2))) +
                        _logmh(jnp.kron(jnp.eye(2), self._msg_down_row[r, c]))
                    )))

        for c in range(self._lat_ham.numcols):
            for r in range(self._lat_ham.numrows - 1):
                self.beliefs_col = \
                    self.beliefs_col.at[c, r].set(_normalise(linalg.expm(
                        self._lat_ham.get_partial_ham_col(c, r) +
                        _logmh(jnp.kron(self._msg_up_col[c, r], jnp.eye(2))) +
                        _logmh(jnp.kron(jnp.eye(2), self._msg_down_col[c, r]))
                    )))
