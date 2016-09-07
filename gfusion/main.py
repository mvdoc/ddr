"""
Densify a known association matrix given similarity matrices for source 1
and source 2.

See Zhang, Ping, Fei Wang, and Jianying Hu. "Towards drug repositioning:
a unified computational framework for integrating multiple aspects of
drug similarity and disease similarity." AMIA Annu Symp Proc. 2014.
"""


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


def _initialize_values(D, S):
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
    pass


def _compute_symmetric_nonnegative_factorization(M):
    """

    Parameters
    ----------
    M : (n, n) symmetric matrix

    Returns
    -------
    S : (n, n) non negative symmetric 

    """
    pass

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
