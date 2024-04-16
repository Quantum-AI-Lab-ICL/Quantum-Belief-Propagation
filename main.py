from hamiltonian import Hamiltonian
from pauli import Pauli
from propagation import BeliefPropagator


if __name__ == "__main__":
    hamiltonian = Hamiltonian(3)
    hamiltonian.set_param_single(0, Pauli.X, 0.5)
    hamiltonian.set_param_single(1, Pauli.X, 0.5)
    hamiltonian.set_param_single(2, Pauli.X, 0.5)
    hamiltonian.set_param_double(0, Pauli.Z, Pauli.Z, 0.5)
    hamiltonian.compute_partial_hamiltonians()
    print(hamiltonian.ham_single)
    print(hamiltonian.ham_double)
    bp = BeliefPropagator(hamiltonian, 0)
    print(0, bp.beliefs)
    bp.step()
    print(1, bp.beliefs)
    bp.step()
    print(2, bp.beliefs)
