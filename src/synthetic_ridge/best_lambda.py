import numpy as np
import copy
from scipy import stats
import math
from scipy.linalg import toeplitz

LAMBDA_GRID = [
    0.0001, 
    0.001,     
    0.01,     
    0.1,       
    1,          
    10,         
    100,                      
]

def quadratic_formula(a, b, c):
    """Solve quadratic equation ax² + bx + c = 0."""
    disc = b**2 - 4 * a * c
    x1 = (-b - math.sqrt(disc)) / (2 * a)
    x2 = (-b + math.sqrt(disc)) / (2 * a)
    return (x1, x2)

def compute_optimal_eta(m_1, m_2, sigma_1, sigma_2):
    """Compute optimal classification threshold eta between two Gaussian distributions."""
    A = -1/sigma_1 + 1/sigma_2
    B = 2*(-m_2/sigma_2 + m_1/sigma_1)
    C = m_2**2/sigma_2 - m_1**2/sigma_1 + np.log(sigma_2/sigma_1)

    if np.abs(A) < 1e-5:
        eta = -C/B  # reduces to the root of a linear equation
    else:
        intersection_1, intersection_2 = quadratic_formula(A, B, C)
        if (intersection_1 - m_1) * (intersection_1 - m_2) <= 0:
            eta = intersection_1
        else:
            eta = intersection_2

    return eta

def fixed_point(cov, means, n_vec_train, lam):
    """Perform fixed-point iteration to compute delta and Qbar."""
    n = np.sum(n_vec_train)
    c_vec = n_vec_train / n
    C1 = cov[0] + np.outer(means[0], means[0])
    C2 = cov[1] + np.outer(means[1], means[1])

    j = 0
    n_iter = 1000
    delta_old = np.ones((2,))
    delta_new = np.zeros((2,))
    delta = np.zeros((2,))
    epsi = 1e-6

    while np.linalg.norm(delta_new - delta_old) > epsi and j < n_iter:
        j = j + 1
        delta_old = copy.deepcopy(delta_new)
        Qbar = np.linalg.inv(c_vec[0]*C1/(1+delta_old[0]) + c_vec[1]*C2/(1+delta_old[1]) + lam*np.eye(C1.shape[1]))
        delta_new[0] = (1/n) * np.trace(C1 @ Qbar)
        delta_new[1] = (1/n) * np.trace(C2 @ Qbar)

    Qbar = np.linalg.inv(c_vec[0]*C1/(1+delta_new[0]) + c_vec[1]*C2/(1+delta_new[1]) + lam*np.eye(C1.shape[1]))
    delta[0] = (1/n) * np.trace(C1 @ Qbar)
    delta[1] = (1/n) * np.trace(C2 @ Qbar)

    return delta, Qbar

def helping_function(cov, means, n_vec_train, lam):
    """Compute derived quantities needed for theoretical error calculation."""
    n = np.sum(n_vec_train)
    c_vec = n_vec_train / n
    C1 = cov[0] + np.outer(means[0], means[0])
    C2 = cov[1] + np.outer(means[1], means[1])

    delta, Qbar = fixed_point(cov, means, n_vec_train, lam)

    M_delta = np.diag(1/(1+delta)) @ means
    A_tilde = np.diag([c_vec[0]/(1+delta[0])**2, c_vec[1]/(1+delta[1])**2])
    V_tilde = (1/n) * np.array([
        [np.trace(C1 @ Qbar @ C1 @ Qbar), np.trace(C1 @ Qbar @ C2 @ Qbar)], 
        [np.trace(C2 @ Qbar @ C1 @ Qbar), np.trace(C2 @ Qbar @ C2 @ Qbar)]
    ])

    t_bar_1 = np.array([np.trace(C1 @ Qbar @ cov[0] @ Qbar)/n, np.trace(C2 @ Qbar @ cov[0] @ Qbar)/n])
    t_bar_2 = np.array([np.trace(C1 @ Qbar @ cov[1] @ Qbar)/n, np.trace(C2 @ Qbar @ cov[1] @ Qbar)/n])
    d1 = np.linalg.inv(np.eye(2) - V_tilde @ A_tilde) @ t_bar_1
    d2 = np.linalg.inv(np.eye(2) - V_tilde @ A_tilde) @ t_bar_2

    K1 = (Qbar @ cov[0] @ Qbar + 
          (c_vec[1]*d1[0]/(1+delta[0])**2) * Qbar @ C1 @ Qbar + 
          (c_vec[0]*d1[1]/(1+delta[1])**2) * Qbar @ C2 @ Qbar)
    K2 = (Qbar @ cov[1] @ Qbar + 
          (c_vec[1]*d2[0]/(1+delta[0])**2) * Qbar @ C1 @ Qbar + 
          (c_vec[0]*d2[1]/(1+delta[1])**2) * Qbar @ C2 @ Qbar)

    deltap1 = [(1/n)*np.trace(cov[0] @ K1), (1/n)*np.trace(cov[1] @ K1)]
    deltap2 = [(1/n)*np.trace(cov[0] @ K2), (1/n)*np.trace(cov[1] @ K2)]
    M_deltap1 = np.diag(deltap1/(1+delta)**2) @ means
    M_deltap2 = np.diag(deltap2/(1+delta)**2) @ means

    return delta, Qbar, K1, K2, M_delta, M_deltap1, M_deltap2

def theory(cov, means, n_vec_train, y_train, lam):
    """Compute theoretical test error for ridge regression using RMT."""
    n = np.sum(n_vec_train)
    J = np.zeros((n, 2))
    J[:n_vec_train[0], 0] = np.ones((n_vec_train[0],))
    J[n_vec_train[0]:n, 1] = np.ones((n_vec_train[1],))

    delta, Qbar, K1, K2, M_delta, M_deltap1, M_deltap2 = helping_function(cov, means, n_vec_train, lam)

    m_th = y_train.T @ J @ M_delta @ Qbar @ means.T / n
    v1 = np.hstack((
        (np.trace(cov[0] @ K1)/((1+delta[0])**2)) * np.ones((n_vec_train[0],)), 
        (np.trace(cov[1] @ K1)/((1+delta[1])**2)) * np.ones((n_vec_train[1],))
    ))
    v2 = np.hstack((
        (np.trace(cov[0] @ K2)/((1+delta[0])**2)) * np.ones((n_vec_train[0],)), 
        (np.trace(cov[1] @ K2)/((1+delta[1])**2)) * np.ones((n_vec_train[1],))
    ))
    V1 = np.diag(v1)
    V2 = np.diag(v2)

    variance_th = np.zeros((2,))
    variance_th[0] = (y_train.T @ V1 @ y_train + 
                     y_train.T @ J @ M_delta @ K1 @ M_delta.T @ J.T @ y_train - 
                     2 * y_train.T @ J @ M_deltap1 @ Qbar @ M_delta.T @ J.T @ y_train)
    variance_th[1] = (y_train.T @ V2 @ y_train + 
                     y_train.T @ J @ M_delta @ K2 @ M_delta.T @ J.T @ y_train - 
                     2 * y_train.T @ J @ M_deltap2 @ Qbar @ M_delta.T @ J.T @ y_train)
    variance_th = variance_th / (n**2)

    eta = compute_optimal_eta(m_th[0], m_th[1], variance_th[0], variance_th[1])

    m_min, m_max = np.sort(np.array([m_th[0], m_th[1]]))
    if m_min == m_th[0]:
        var_min = variance_th[0]
        var_max = variance_th[1]
    else:
        var_min = variance_th[1]
        var_max = variance_th[0]

    prob_1 = stats.norm.cdf(eta, loc=m_max, scale=np.sqrt(var_max))
    prob_2 = 1 - stats.norm.cdf(eta, loc=m_min, scale=np.sqrt(var_min))

    error_test_th = 0.5 * prob_1 + 0.5 * prob_2

    return error_test_th

def find_best_lambda(means, cov, n_vec):
    """
    Finds the lambda that minimizes the theoretical test error.
    
    Parameters:
    means: list of mean vectors (2 x d)
    cov: list of covariance matrices (2 x d x d)
    n_vec: list containing the number of samples in each class [n1, n2]
    
    Returns:
    best_lambda: lambda that minimizes theoretical test error
    all_lambdas: grid of lambda values tested
    all_errors: corresponding error_test_th for each lambda
    """
    y_train = -np.ones((sum(n_vec), 1))
    y_train[:n_vec[0]] = 1  # Class 1 labeled as 1, class 2 as -1

    lambda_grid = np.array(LAMBDA_GRID)
    errors = []

    for lam in lambda_grid:
        try:
            err = theory(cov, means, n_vec, y_train[:, 0], lam)
        except Exception as e:
            err = np.inf  # in case numerical errors occur
        errors.append(err)

    errors = np.array(errors)
    best_idx = np.argmin(errors)
    best_lambda = lambda_grid[best_idx]

    return best_lambda, lambda_grid, errors



if __name__ == "__main__":
    d = 5
    eps = np.random.uniform(low=0, high=2, size=d)
    mu1 = np.ones((d,))
    mu2 = -eps * mu1
    means = np.array([mu1, mu2])
    alpha1 = np.random.uniform(low=0, high=0.9, size=1)
    alpha2 = np.random.uniform(low=0, high=0.9, size=1)
    cov = [toeplitz(alpha1**(np.arange(d))), toeplitz(alpha2**(np.arange(d)))]
    n_vec = [100, 200]

    best_lambda, lambda_grid, errors = find_best_lambda(means, cov, n_vec)
    print("Best lambda:", best_lambda, "with error:", errors[np.argmin(errors)])

    