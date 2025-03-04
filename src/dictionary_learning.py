import numpy as np
from sklearn.decomposition import NMF, MiniBatchNMF


def train_dictionary(
    feature: np.ndarray,
    num_atom: int,
    batch_size: int = -1,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train a dictionary using Non-negative Matrix Factorization (NMF).

    This function factorizes the input feature matrix into a coefficient matrix (Z) and
    a dictionary component matrix (D) using either standard NMF or MiniBatchNMF, depending
    on the specified batch size. The algorithm minimizes the reconstruction error while ensuring
    non-negativity of the matrices.

    Args:
        feature (np.ndarray): A 2D array of shape (n_samples, n_features) representing the input data.
        num_atom (int): The number of dictionary atoms (components) to extract.
        batch_size (int, optional): The batch size for MiniBatchNMF. If set to -1, the standard NMF
            is used instead. Defaults to -1.
        tol (float, optional): The tolerance for the stopping condition of the algorithm.
            Defaults to 1e-4.
        max_iter (int, optional): The maximum number of iterations to run the algorithm.
            Defaults to 1000.

    Returns:
        tuple: A tuple containing:
            - Z (np.ndarray): The coefficient matrix obtained from the transformation.
            - D (np.ndarray): The dictionary component matrix.
            - reconstruction_err (float): The reconstruction error of the NMF model after training.

    Raises:
        AssertionError: If the input feature is not a 2D array or if the resulting matrices contain negative values.

    """
    assert feature.ndim == 2

    if batch_size == -1:
        nmf_model = NMF(
            n_components=num_atom,
            tol=tol,
            max_iter=max_iter,
            verbose=True,
        )
    else:
        nmf_model = MiniBatchNMF(
            n_components=num_atom,
            tol=tol,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=True,
        )

    Z = nmf_model.fit_transform(feature)  # noqa: N806
    D = nmf_model.components_  # noqa: N806
    assert (Z >= 0).all()
    assert (D >= 0).all()
    return Z, D, nmf_model.reconstruction_err_
