from jax.scipy import linalg
import jax.numpy as jnp
import matplotlib.pyplot as plt

from examples.example_utils import rdm, matrix_3x3, lat_ham_setup
from lattice_propagation import LatticeBeliefPropagator


if __name__ == "__main__":
    x_coef = -3
    zz_coef = -1
    size = 3
    errors = []
    space = jnp.linspace(0, 2, 20, dtype=jnp.float32)

    for beta in space:
        lat_ham = lat_ham_setup(size, beta, x_coef, 0, zz_coef)

        H = matrix_3x3(x_coef, zz_coef)
        rho = linalg.expm(-beta * H)
        rho /= jnp.trace(rho)

        exact_sol = jnp.zeros((size, size, 2, 2), dtype=jnp.complex64)
        for r in range(size):
            for c in range(size):
                exact_sol = exact_sol.at[r, c].set(rdm(rho, 1, r * size + c))

        propagator = LatticeBeliefPropagator(lat_ham)
        for i in range(size * size):
            propagator.step()
            total_error = 0.0
            for r in range(size):
                for c in range(size):
                    total_error += jnp.linalg.norm(
                        propagator.mean_single_belief(r, c) -
                        exact_sol[r, c]
                    )
        errors.append(total_error / (size * size))

    plt.plot(space, errors)
    plt.xlabel("beta")
    plt.ylabel("average norm of error")
    plt.title("Error against exact solution in 3x3 matrices by beta value")
    plt.savefig("examples/results/2d/error_3x3.png")
