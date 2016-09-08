"""
Densify a known association matrix given similarity matrices for source 1
and source 2.

See Zhang, Ping, Fei Wang, and Jianying Hu. "Towards drug repositioning:
a unified computational framework for integrating multiple aspects of
drug similarity and disease similarity." AMIA Annu Symp Proc. 2014.
"""

import numpy as np

def densify(D, S, R):
    """Densify a matrix R

    Arguments
    ---------
    D : np.ndarray (n_similarities, n_features)
        flattened upper triangular similarity matrices for source 1

    S : np.ndarray (m_similarities, m_features)
        flattened upper triangular similarity matrices for source 2

    R : np.ndarray (n_features, m_features)
        known association matrix between source 1 and source 2

    Returns
    -------
    Theta : np.ndarray (n_features, m_features)
        densified association matrix between source 1 and source 2
    """
    pass


def _initialize_values(D, S, lambda1 = 1, lambda2 = 1, delta1 = 1, delta2 = 1):
    """Initialize values

    Arguments
    ---------
    D : np.ndarray (n_similarities, n_features)
    similarity matrices for source 1

    S : np.ndarray (m_similarities, m_features)
        similarity matrices for source 2

    Returns
    -------
    lambda1 : float
    lambda2 : float
    delta1 : float
    delta2 : float
    omega : np.ndarray (n_similarities, 1)
    pi : np.ndarray (m_similarities, 1)
    """
    Kd = D.shape(0)
    omega = np.ones((Kd,1))/Kd

    Ks = S.shape(0)
    pi = np.ones((Ks,1))/Ks

    return {'lambda1':lambda1, 'lambda2':lambda2, 'delta1':delta1, 'delta2':delta2, 'omega':omega, 'pi':pi}


def _compute_symnmf(X, k, Hinit = np.array([]), maxiter = 10000, tol = 1e-4, sigma = 0.1, beta = 0.1):
    """

    https://github.com/andybaoxv/symnmf
    Da Kuang, Chris Ding, Haesun Park,
    Symmetric Nonnegative Matrix Factorization for Graph Clustering,
    The 12th SIAM International Conference on Data Mining (SDM '12), pp. 106--117.
    Parameters
    ----------
    X : (n, n) symmetric matrix
    k : number of clusters
    Hinit : (n, k) initialization of H
    maxiter : max number of iterations
    tol : tolerance parameter to determine convergence
    sigma : acceptance parameter
    beta : reduction factor to decrease step size of gradient search

    Returns
    -------
    H : (n, n) non negative symmetric 
    iter : number of iterations used
    energy: f(H)
    """
    
    #should implement check that X is symmetric (at least square)
    #should implement checks for optional input

    n = X.shape(0)

    if Hinit.size < 1:
        H = 2 * np.sqrt( np.mean(A) / k) * np.random.rand(n, k)
    else
        H = Hinit


    projnorm_idx = np.zeros([n,k])
    R = np.zeros([1,k])
    p = np.zeros([1,k])
    energy = np.square( np.linalg.norm(A - np.dot(H,H.T), 'fro') )
    gradH = 4 * (np.dot(H, np.dot(H.T,H)) - np.dot(A,H))
    initgrad = np.linalg.norm(gradH, 'fro')

    for iter in xrange(1,maxiter):
        gradH = 4*(np.dot(H,np.dot(H.T,H)) - np.dot(A,H))
        projnorm_idx_prev = projnorm_idx
        projnorm_idx = gradH<=eps | H>eps #
        projnorm = np.linalg.norm(gradH(projnorm_idx)) #

    return {'H':H, 'iter':iter, 'energy':energy}


def _initialize_latent_matrices(D, S, omega, pi):
    """Performs Symmetric Nonnegative Matrix Factorization to initialize
    the latent matrices U and V

    Arguments
    ---------
    D : np.ndarray (n_similarities, n_features)
        similarity matrices for source 1

    S : np.ndarray (m_similarities, m_features)
        similarity matrices for source 2

    omega : np.ndarray (n_similarities, 1)

    pi : np.ndarray (m_similarities, 1)

    Returns
    -------
    U : np.ndarray (n_features, latent_dimension1)
        latent matrix for source 1

    V : np.ndarray (m_features, latent_dimension2)
        latent matrix for source 2
    """
    pass


def _solve_lambda(U, V):
    """Solve lambda and all my problems

    Arguments
    ---------
    U : np.ndarray (n_features, latent_dimension1)

        latent matrix for source 1

    V : np.ndarray (m_features, latent_dimension2)

        latent matrix for source 2

    Returns
    -------
    L : np.ndarray (latent_dimension1, latent_dimension2)

        aka lambda, something we're trying to optimize
    """
    pass


def _solve_theta(U, V, R, L):
    """Solve theta

    Arguments
    ---------
    U : np.ndarray (n_features, latent_dimension1)

        latent matrix for source 1

    V : np.ndarray (m_features, latent_dimension2)

        latent matrix for source 2

    R : np.ndarray (n_features, m_features)

        known association matrix between source 1 and source 2

    L : np.ndarray (latent_dimension1, latent_dimension2)


    Returns
    -------
    Theta : np.ndarray (n_features, m_features)
        densified association matrix between source 1 and source 2
    """
    pass


def _solve_omega(D, U, delta1, omega):
    """Solve omega

    Arguments
    ---------
    D : np.ndarray (n_similarities, n_features)
        similarity matrices for source 1

    U : np.ndarray (n_features, latent_dimension1)

        latent matrix for source 1

    delta1 : float

    omega : np.ndarray (n_similarities, 1)

    Returns
    -------
    omega : np.ndarray (n_similarities, 1)
    """
    pass


def _solve_pi(S, V, delta2, pi):
    """Solve pi

    Arguments
    ---------
    S : np.ndarray (m_similarities, m_features)
        similarity matrices for source 2

    V : np.ndarray (m_features, latent_dimension2)

        latent matrix for source 2

    delta2 : float

    pi : np.ndarray (m_similarities, 1)

    Returns
    -------
    pi : np.ndarray (m_similarities, 1)
    """
    pass


def _solve_u(Theta, U, V, L, lambda1, D, omega):
    """Solve U
    Arguments
    ---------

    TODO

    Returns
    -------
    U : np.ndarray (n_features, latent_dimension1)

        latent matrix for source 1
    """
    pass


def _solve_v(Theta, U, V, L, lambda2, S, pi):
    """Solve V
    Arguments
    ---------

    TODO

    Returns
    -------
    V : np.ndarray (m_features, latent_dimension2)

        latent matrix for source 2
    """
    pass
