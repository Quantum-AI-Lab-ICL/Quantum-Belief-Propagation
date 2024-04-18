import jax.numpy as jnp

from hamiltonian import Hamiltonian
from pauli import Pauli
from propagation import BeliefPropagator
from utils import _double_to_single_trace


if __name__ == "__main__":
    hamiltonian = Hamiltonian(3)
    hamiltonian.set_param_single(0, Pauli.X, 0.5)
    hamiltonian.set_param_single(1, Pauli.X, 0.5)
    hamiltonian.set_param_single(2, Pauli.X, 0.5)
    # hamiltonian.set_param_double(0, Pauli.Z, Pauli.Z, 0.5)
    # hamiltonian.set_param_double(1, Pauli.Z, Pauli.Z, 0.5)
    hamiltonian.compute_partial_hamiltonians()
    print(hamiltonian.ham_single)
    print(hamiltonian.ham_double)
    bp = BeliefPropagator(hamiltonian, 0)

    jnp.set_printoptions(precision=2, suppress=True, linewidth=1000)
    for i in range(5):
        bp.step()
        print(i)
        print(bp.beliefs)

    print(_double_to_single_trace(bp.beliefs[0], 0))
    print(_double_to_single_trace(bp.beliefs[1], 1))
