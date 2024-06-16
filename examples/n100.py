import jax.numpy as jnp
import matplotlib.pyplot as plt

from propagation import BeliefPropagator
from examples.example_utils import get_single_rho, trans_mag, \
    correlation, ham_setup


def main():
    N = 100
    Mx = []
    Cxx = []
    HN = []
    for beta in jnp.linspace(0.1, 2.0, 20, dtype=jnp.float32):
        print(beta)
        ham = ham_setup(N, beta, -1.05, 0.5, 1.0)
        bp = BeliefPropagator(ham, 1)
        for i in range(N):
            print(i)
            bp.step()
        results = get_single_rho(bp.beliefs, N)
        Mx.append(trans_mag(results))
        Cxx.append(correlation(bp.beliefs))
        HN.append(energy_expectation(
            bp.beliefs, ham.hamiltonians, ham.beta) / N)

    plt.scatter(HN, Mx)
    plt.xlabel("<H>/N")
    plt.ylabel("Mx")
    plt.savefig("examples/results/n100_Mx.png")
    plt.clf()

    plt.scatter(HN, Cxx)
    plt.xlabel("<H>/N")
    plt.ylabel("Cxx")
    plt.savefig("examples/results/n100_Cxx.png")
    plt.clf()


def energy_expectation(double_rho, partial_ham, beta):
    result = 0.0
    for i in range(double_rho.shape[0]):
        result += jnp.trace(double_rho[i] @ partial_ham[i] / (-beta))
    return result


if __name__ == "__main__":
    main()
