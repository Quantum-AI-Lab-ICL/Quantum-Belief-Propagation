import jax.numpy as jnp
import jax.typing


FAILED_MSG = "Verification failed: "


def _verify_hermitian(M: jax.typing.ArrayLike):
    """
    Verify that the input jax array is Hermitian. Raise a ValueError if the
    verfication fails.

    Parameters:
        M (jax.typing.ArrayLike): input jax array to verify
    """

    if not jnp.allclose(jnp.conjugate(jnp.transpose(M)), M):
        error_msg = f"{FAILED_MSG}The input is not Hermitian."
        raise ValueError(error_msg)
