import jax.numpy as jnp
import matplotlib.pyplot as plt

from lattice_hamiltonian import LatticeHamiltonian
from lattice_propagation import LatticeBeliefPropagator
from pauli import Pauli
from examples.example_utils import get_single_rho, trans_mag, \
    correlation_spin_2d


def main():
    N = 3
    Mx = []
    Czz = []
    HN = []
    for beta in jnp.linspace(0.1, 2.0, 20, dtype=jnp.float32):
        print(beta)
        ham = ham_setup(N, beta, -2.5, -1.0)
        bp = LatticeBeliefPropagator(ham)
        for i in range(N * N):
            print(i)
            bp.step()
        single_results = single_rho(bp, N)
        Mx.append(trans_mag(single_results))
        Czz.append((correlation_spin_2d(bp.beliefs_row) +
                   correlation_spin_2d(bp.beliefs_col)) / 2)
        HN.append((
            energy_expectation(bp.beliefs_row, ham.hams_row, ham.beta) +
            energy_expectation(bp.beliefs_col, ham.hams_col, ham.beta)
        ) / 2)

    print(Mx)
    print(Czz)
    print(HN)
    plt.scatter(HN, Mx)
    plt.xlabel("<H>/N")
    plt.ylabel("Mx")
    plt.savefig("examples/results/n10x10_Mx.png")
    plt.clf()

    plt.scatter(HN, Czz)
    plt.xlabel("<H>/N")
    plt.ylabel("Czz")
    plt.savefig("examples/results/n10x10_Czz.png")
    plt.clf()


def ham_setup(N, beta, x_coef, zz_coef):
    ham = LatticeHamiltonian(N, N, beta)
    for r in range(N):
        for c in range(N):
            ham.set_param_single(r, c, Pauli.X, x_coef)
    for r in range(N):
        for c in range(N - 1):
            ham.set_param_double_row(r, c, Pauli.Z, Pauli.Z, zz_coef)
    for c in range(N):
        for r in range(N - 1):
            ham.set_param_double_row(c, r, Pauli.Z, Pauli.Z, zz_coef)
    ham.compute_partial_hamiltonians()
    return ham


def single_rho(lbp: LatticeBeliefPropagator, N):
    result = jnp.zeros((N * N,
                        2, 2), dtype=jnp.complex64)
    for r in range(N):
        for c in range(N):
            result = result.at[r * N + c].set(
                lbp.mean_single_belief(r, c))
    return result


def energy_expectation(double_rho, partial_ham, beta):
    result = 0.0
    for i in range(double_rho.shape[0]):
        for j in range(double_rho.shape[1]):
            result += jnp.trace(double_rho[i, j] @ partial_ham[i, j] / (-beta))
    return result / (double_rho.shape[0] * double_rho.shape[1])


if __name__ == "__main__":
    main()
