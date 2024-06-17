import jax.numpy as jnp
import matplotlib.pyplot as plt

from examples.example_utils import trans_mag, correlation_spin_2d, \
    energy_expectation, lat_ham_setup, square_lattice_single_beliefs
from lattice_propagation import LatticeBeliefPropagator


def main():
    N = 4
    Mx = []
    Czz = []
    HN = []
    reg_factor = 0.01
    for beta in jnp.linspace(0.1, 2.0, 20, dtype=jnp.float32):
        ham = lat_ham_setup(N, beta, -2.5, 0, -1.0)
        bp = LatticeBeliefPropagator(ham, reg_factor)
        for i in range(N * N):
            bp.step()
        single_results = square_lattice_single_beliefs(bp, N)
        Mx.append(trans_mag(single_results))
        Czz.append((
            correlation_spin_2d(bp.beliefs_row) +
            correlation_spin_2d(bp.beliefs_col)) / 2)
        HN.append((
            energy_expectation(bp.beliefs_row, ham.hams_row, ham.beta) +
            energy_expectation(bp.beliefs_col, ham.hams_col, ham.beta)
        ) / 2)

    print(HN)
    print(Mx)
    plt.scatter(HN, Mx)
    plt.title(f"Transverse magnetisation for {N}x{N} system")
    plt.xlabel("<H>/N")
    plt.ylabel("Mx")
    plt.savefig(f"examples/results/n{N}x{N}_Mx_{reg_factor}.png")
    plt.clf()

    plt.scatter(HN, Czz)
    plt.title(f"Spin-spin correlation for {N}x{N} system")
    plt.xlabel("<H>/N")
    plt.ylabel("Czz")
    plt.savefig(f"examples/results/n{N}x{N}_Czz_{reg_factor}.png")
    plt.clf()


if __name__ == "__main__":
    main()
