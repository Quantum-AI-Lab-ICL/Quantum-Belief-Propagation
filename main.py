from pauli import Pauli
from hamiltonian import Hamiltonian


if __name__ == "__main__":
    hamiltonian = Hamiltonian(2)
    hamiltonian.set_param_single(0, Pauli.X, 0.5)
    hamiltonian.set_param_single(1, Pauli.X, 0.5)
    hamiltonian.set_param_double(0, Pauli.Z, Pauli.Z, 0.5)
    hamiltonian.compute_partial_hamiltonians()
    print(hamiltonian.ham_single)
    print(hamiltonian.ham_double)
