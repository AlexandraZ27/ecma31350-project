"""Accurate short Gaussian rules for fitted mixture densities."""
import numpy as np


def mixture_gauss(density, hermite_quadrature, order=32):
    """Stieltjes/Golub-Welsch rule for each fitted Gaussian mixture.

    Orthogonalize against a 64-point/component Hermite reference measure.
    This reduces only numerical integration cost, not the fitted model class.
    """
    a, q = hermite_quadrature(density, 64)
    mu = (a*q).sum(axis=1)
    sd = np.sqrt(((a-mu[:, None])**2*q).sum(axis=1))
    x = (a-mu[:, None])/sd[:, None]
    old = np.zeros_like(x); p = np.ones_like(x); beta = np.zeros(len(x))
    diagonal = []; off = []
    for j in range(order):
        alpha = (q*x*p*p).sum(axis=1); diagonal.append(alpha)
        if j == order-1: break
        r = (x-alpha[:, None])*p-beta[:, None]*old
        beta = np.sqrt((q*r*r).sum(axis=1)); off.append(beta)
        old, p = p, r/beta[:, None]
    jacobi = np.zeros((len(x), order, order))
    idx = np.arange(order)
    jacobi[:, idx, idx] = np.array(diagonal).T
    idx = np.arange(order-1)
    jacobi[:, idx, idx+1] = jacobi[:, idx+1, idx] = np.array(off).T
    values, vectors = np.linalg.eigh(jacobi)
    nodes = mu[:, None] + sd[:, None]*values
    weights = vectors[:, 0, :]**2
    assert np.allclose(weights.sum(axis=1), 1, atol=1e-12)
    assert np.allclose((weights*nodes).sum(axis=1), mu, atol=1e-10)
    assert np.allclose((weights*nodes**2).sum(axis=1), (q*a*a).sum(axis=1), atol=1e-9)
    return nodes, weights
