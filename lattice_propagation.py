"""
LatticeBeliefPropagator class for running the 2D algorithm.
"""
from jax.scipy import linalg
import jax
import jax.numpy as jnp

from const import MATRIX_SIZE_DOUBLE, MATRIX_SIZE_SINGLE
from lattice_hamiltonian import LatticeHamiltonian
from utils import _double_to_single_trace, _logmh, _normalise


def _checked_logmh_inv(msgs, ix0, ix1, reg_factor=0.01):
    """
    Calculate the logarithm of the inverse of a message at the index if it is
    valid.
    """

    if 0 <= ix0 and ix0 < msgs.shape[0] \
            and 0 <= ix1 and ix1 < msgs.shape[1]:

        msg = msgs[ix0, ix1]
        cond = jnp.linalg.cond(msg)
        delta = reg_factor / msg
        if cond > 0.1 / reg_factor:
            msg = msg + delta * jnp.eye(2)

        return _logmh(linalg.inv(msg))
    return jnp.zeros((2, 2), dtype=jnp.complex64)


def _checked_logmh_trace(beliefs, ix0, ix1, trace_id):
    """
    Calculate the partial trace of the belief at the index if it is valid.
    """
    if 0 <= ix0 and ix0 < beliefs.shape[0] \
            and 0 <= ix1 and ix1 < beliefs.shape[1]:
        return _logmh(_double_to_single_trace(beliefs[ix0, ix1], trace_id))
    return jnp.zeros((2, 2), dtype=jnp.complex64)


class LatticeBeliefPropagator:
    """
    Class for performing the quantum belief propagation algorithm on a 2D
    lattice.

    Attributes
    ----------
    lat_ham: LatticeHamiltonian
        the Hamiltonian of the underlying quantum system
    beliefs: jax.Array
        the pair-wise beliefs for the reduced density matrices
    reg_factor: jnp.float32
        the regularisation factors for regularising messages -
        default value 0.01

    Methods
    -------
    step
        run one step of the algorithm as specified, including the computation
        of the messages and the new beliefs
    """

    def __init__(self, lat_ham: LatticeHamiltonian,
                 reg_factor: jnp.float32 = 0.01):
        """
        TODO
        """

        self._lat_ham = lat_ham
        self.reg_factor = reg_factor
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
        Run one step of the algorithm as specified. This includes the
        computation of the messages and the new beliefs.
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

        for r in range(self._lat_ham.numrows):
            for c in range(self._lat_ham.numcols - 1):
                new_msg_up_row = new_msg_up_row.at[r, c].set(
                    self._compute_new_msg_up_row(r, c))
                new_msg_down_row = new_msg_down_row.at[r, c].set(
                    self._compute_new_msg_down_row(r, c))

        for c in range(self._lat_ham.numcols):
            for r in range(self._lat_ham.numrows - 1):
                new_msg_up_col = new_msg_up_col.at[c, r].set(
                    self._compute_new_msg_up_col(c, r))
                new_msg_down_col = new_msg_down_col.at[c, r].set(
                    self._compute_new_msg_down_col(c, r))

        self._msg_up_row = new_msg_up_row
        self._msg_down_row = new_msg_down_row
        self._msg_up_col = new_msg_up_col
        self._msg_down_col = new_msg_down_col

        self._compute_new_beliefs()

    def _compute_new_msg_up_row(self, r, c):
        """Compute messages up along the row direction."""
        return _normalise(linalg.expm(
            _checked_logmh_trace(self.beliefs_row, r, c-1, 0) +
            _checked_logmh_inv(self._msg_down_row, r, c-1, self.reg_factor) +
            _checked_logmh_trace(self.beliefs_col, c, r-1, 0) +
            _checked_logmh_inv(self._msg_down_col, c, r-1, self.reg_factor) +
            _checked_logmh_trace(self.beliefs_col, c, r, 1) +
            _checked_logmh_inv(self._msg_up_col, c, r, self.reg_factor)
        ))

    def _compute_new_msg_down_row(self, r, c):
        """Compute messages down along the row direction."""
        return _normalise(linalg.expm(
            _checked_logmh_trace(self.beliefs_row, r, c+1, 1) +
            _checked_logmh_inv(self._msg_up_row, r, c+1, self.reg_factor) +
            _checked_logmh_trace(self.beliefs_col, c+1, r, 1) +
            _checked_logmh_inv(self._msg_up_col, c+1, r, self.reg_factor) +
            _checked_logmh_trace(self.beliefs_col, c+1, r-1, 0) +
            _checked_logmh_inv(self._msg_down_col, c+1, r-1, self.reg_factor)
        ))

    def _compute_new_msg_up_col(self, c, r):
        """Compute messages up along the column direction."""
        return _normalise(linalg.expm(
            _checked_logmh_trace(self.beliefs_col, c, r-1, 0) +
            _checked_logmh_inv(self._msg_down_col, c, r-1, self.reg_factor) +
            _checked_logmh_trace(self.beliefs_row, r, c-1, 0) +
            _checked_logmh_inv(self._msg_down_row, r, c-1, self.reg_factor) +
            _checked_logmh_trace(self.beliefs_row, r, c, 1) +
            _checked_logmh_inv(self._msg_up_row, r, c, self.reg_factor)
        ))

    def _compute_new_msg_down_col(self, c, r):
        """Compute messages down along the column direction."""
        return _normalise(linalg.expm(
            _checked_logmh_trace(self.beliefs_col, c, r+1, 1) +
            _checked_logmh_inv(self._msg_up_col, c, r+1, self.reg_factor) +
            _checked_logmh_trace(self.beliefs_row, r+1, c, 1) +
            _checked_logmh_inv(self._msg_up_row, r+1, c, self.reg_factor) +
            _checked_logmh_trace(self.beliefs_row, r+1, c-1, 0) +
            _checked_logmh_inv(self._msg_down_row, r+1, c-1, self.reg_factor)
        ))

    def _compute_new_beliefs(self):
        """Compute new beliefs using the messages."""
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

    def mean_single_belief(self, rowindex: jnp.int32,
                           colindex: jnp.int32) -> jax.Array:
        acc = jnp.zeros((2, 2), dtype=jnp.complex64)
        count = 0
        for c, trace_id in [(colindex - 1, 0), (colindex, 1)]:
            if 0 <= c and c < self.beliefs_row.shape[1]:
                acc += _double_to_single_trace(
                    self.beliefs_row[rowindex, c], trace_id)
                count += 1
        for r, trace_id in [(rowindex - 1, 0), (rowindex, 1)]:
            if 0 <= r and r < self.beliefs_col.shape[1]:
                acc += _double_to_single_trace(
                    self.beliefs_col[colindex, r], trace_id)
                count += 1
        return acc / count
