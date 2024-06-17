import matplotlib.pyplot as plt
import time

from examples.example_utils import get_single_rho, ham_setup
from propagation import BeliefPropagator


def main():
    beta = 1
    x_coeff = -1.05
    z_coeff = 0.5
    zz_coeff = 1.0

    # warm up
    for _ in range(5):
        size = 14
        ham = ham_setup(size, beta, x_coeff, z_coeff, zz_coeff)

        bp_start_time = time.perf_counter()
        bp = BeliefPropagator(ham)
        for i in range(size):
            bp.step()
        get_single_rho(bp.beliefs, size)

    size_range = range(3, 13)
    bp_times = []

    for size in size_range:
        ham = ham_setup(size, beta, x_coeff, z_coeff, zz_coeff)

        bp_start_time = time.perf_counter()
        bp = BeliefPropagator(ham)
        for i in range(size):
            bp.step()
        get_single_rho(bp.beliefs, size)
        bp_end_time = time.perf_counter()

        bp_times.append(bp_end_time - bp_start_time)

    plt.plot(size_range, bp_times)
    plt.ylabel("Computation time")
    plt.xlabel("Number of particles")
    plt.title("Computation time for belief propagation")
    plt.tight_layout()
    plt.savefig("examples/results/1d/time.png")


if __name__ == "__main__":
    main()
