"""
Hamiltonian class for model setup in the 1D algorithm.
"""

import jax.numpy as jnp

from const import NUM_PARAMS_SINGLE, NUM_PARAMS_DOUBLE, MATRIX_SIZE_SINGLE, \
    MATRIX_SIZE_DOUBLE
from pauli import Pauli, _pauli_matrix, _pauli_matrix_2d, _parse_pauli_index


class Hamiltonian:
    """
    Class representing the Hamiltonian of the quantum system in 1D.

    Attributes
    ----------
    size: jnp.int32
        size of the system, i.e. number of particles
    beta: jnp.float32
        beta coefficient on the Hamiltonian in the thermal state representation
    hamiltonians: jax.Array
        partial Hamiltonian matrices of the system

    Methods
    -------
    set_param_single
        set the model parameter for a component of the Hamiltonian that acts on
        a single particle
    set_param_double
        set the model parameter for a component of the Hamiltonian that acts on
        a pair of neighbouring particles
    compute_partial_hamiltonians
        compute the partial Hamiltonian matrices as specified in the algorithm
    get_partial_hamiltonian
        get the partial Hamiltonian matrix at a particular index
    """

    def __init__(self, size: jnp.int32, beta: jnp.float32):
        """
        Initialise the Hamiltonian class.

        Parameters
        ----------
        size: jnp.int32
            size of the system, i.e. number of particles
        beta: jnp.float32
            beta coefficient on the Hamiltonian in the thermal state
            representation
        """

        self.size = size
        self.beta = beta
        self._params_single = jnp.zeros((size, NUM_PARAMS_SINGLE),
                                        dtype=jnp.complex64)
        self._params_double = jnp.zeros((size - 1, NUM_PARAMS_DOUBLE),
                                        dtype=jnp.complex64)
        self._reset_partial_hamiltonians()

    def set_param_single(self, index: jnp.int32, pauli: Pauli,
                         value: jnp.float32):
        """
        Set the model parameter for a component of the Hamiltonian that acts on
        a single particle.

        Parameters
        ----------
        index: jnp.int32
            index of the particle at which to set the model parameter
        pauli: Pauli
            specification of which Pauli matrix the parameter is for
        value: jnp.float32
            value of the parameter
        """

        pauli_index = pauli.value
        self._params_single = \
            self._params_single.at[index, pauli_index].set(-self.beta * value)

    def set_param_double(self, index: jnp.int32, pauli_0: Pauli,
                         pauli_1: Pauli, value: jnp.complex64):
        """
        Set the model parameter for a component of the Hamiltonian that acts on
        a pair of neighbouring particles.

        Parameters
        ----------
        index: jnp.int32
            index of the first particle of the pair at which to set the model
            parameter
        pauli_0: Pauli
            specification of which Pauli matrix the parameter is for at the
            first particle
        pauli_1: Pauli
            specification of which Pauli matrix the parameter is for at the
            second particle
        value: jnp.float32
            value of the parameter
        """

        pauli_index = _parse_pauli_index(pauli_0, pauli_1)
        self._params_double = \
            self._params_double.at[index, pauli_index].set(-self.beta * value)

    def _reset_partial_hamiltonians(self):
        """
        Reset the partial Hamiltonians.
        """

        self.hamiltonians = jnp.zeros((self.size - 1, MATRIX_SIZE_DOUBLE,
                                       MATRIX_SIZE_DOUBLE),
                                      dtype=jnp.complex64)

    def compute_partial_hamiltonians(self):
        """
        Compute the partial Hamiltonian matrices as specified in the algorithm.
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
        Give correct weighting of component matrices for partial Hamiltonian
        calculation.
        """

        if index == 0 or index == self.size - 1:
            return ham_single[index]
        return ham_single[index] / 2

    def get_partial_hamiltonian(self, index):
        """
        Get the partial Hamiltonian matrix at a particular index.

        Parameters
        ----------
        index: jnp.int32
            index of the first particle of the pair at which to get the partial
            Hamiltonian matrix

        Returns
        -------
        partial_hamiltonian: jax.Array
            the partial Hamiltonian
        """

        return self.hamiltonians[index]
