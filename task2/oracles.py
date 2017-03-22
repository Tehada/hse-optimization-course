import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def minimize_directional(self, x, d):
        """
        Minimizes the function with respect to a specific direction:
            Finds alpha = argmin f(x + alpha d)
        """

        return -np.dot(d, self.grad(x)) / np.dot(d, self.A.dot(d))


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        logreg = scipy.linalg.norm(np.logaddexp(0.0, -self.b * self.matvec_Ax(x)), 1)
        return self.b.size**-1 * logreg + self.regcoef / 2.0 * scipy.linalg.norm(x)**2

    def grad(self, x):
        logreg = self.matvec_ATx(self.b * scipy.special.expit(-self.b * self.matvec_Ax(x)))
        return -self.b.size**-1 * logreg + self.regcoef * x

    def hess(self, x):
        sigma = scipy.special.expit(-self.b * self.matvec_Ax(x))
        logreg = self.matmat_ATsA(sigma * (1.0 - sigma))
        return self.b.size**-1 * logreg + self.regcoef * np.identity(x.size)

    def hess_vec(self, x, d):
        sigma = scipy.special.expit(self.b * self.matvec_Ax(x))
        logreg = self.matvec_ATx(sigma * (1.0 - sigma) * self.matvec_Ax(d))
        return self.b.size**-1 * logreg + self.regcoef * d


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

        self.last_x = None
        self.last_x_directional = None
        self.last_d_directional = None
        self.last_xalphad_directional = None
        self.last_Ax = None
        self.last_Ax_directional = None
        self.last_Ad_directional = None
        self.last_Axalphad_directional = None


    def calculated(self):
        return [
            (self.last_x, self.last_Ax),
            (self.last_x_directional, self.last_Ax_directional),
            (self.last_d_directional, self.last_Ad_directional),
            (self.last_xalphad_directional, self.last_Axalphad_directional),
        ]

    def update_x(self, x):
        for last_x, last_Ax in self.calculated():
            if np.array_equal(last_x, x):
                self.last_x, self.last_Ax = np.copy(last_x), np.copy(last_Ax)
                return
        self.last_x, self.last_Ax = np.copy(x), self.matvec_Ax(x)

    def update_x_directional(self, x):
        for last_x, last_Ax in self.calculated():
            if np.array_equal(last_x, x):
                self.last_x_directional, self.last_Ax_directional = np.copy(last_x), np.copy(last_Ax)
                return
        self.last_x_directional, self.last_Ax_directional = np.copy(x), self.matvec_Ax(x)

    def update_d_directional(self, x):
        for last_x, last_Ax in self.calculated():
            if np.array_equal(last_x, x):
                self.last_d_directional, self.last_Ad_directional = np.copy(last_x), np.copy(last_Ax)
                return
        self.last_d_directional, self.last_Ad_directional = np.copy(x), self.matvec_Ax(x)

    def update_xalphad_directional(self, x, d, alpha):
        x_trial = x + alpha * d

        for last_x, last_Ax in self.calculated():
            if np.array_equal(last_x, x_trial):
                self.last_xalphad_directional, self.last_Axalphad_directional = np.copy(last_x), np.copy(last_Ax)
                return

        self.update_x_directional(x)
        self.update_d_directional(d)

        self.last_xalphad_directional = x_trial
        self.last_Axalphad_directional = self.last_Ax_directional + alpha * self.last_Ad_directional


    def func(self, x):
        self.update_x(x)
        logreg = scipy.linalg.norm(np.logaddexp(0.0, -self.b * self.last_Ax), 1)
        return self.b.size**-1 * logreg + self.regcoef / 2.0 * scipy.linalg.norm(x)**2

    def grad(self, x):
        self.update_x(x)
        logreg = self.matvec_ATx(self.b * scipy.special.expit(-self.b * self.last_Ax))
        return -self.b.size**-1 * logreg + self.regcoef * x

    def hess(self, x):
        self.update_x(x)
        sigma = scipy.special.expit(-self.b * self.last_Ax)
        logreg = self.matmat_ATsA(sigma * (1.0 - sigma))
        return self.b.size**-1 * logreg + self.regcoef * np.identity(x.size)

    def func_directional(self, x, d, alpha):
        self.update_xalphad_directional(x, d, alpha)
        logreg = scipy.linalg.norm(np.logaddexp(0.0, -self.b * self.last_Axalphad_directional), 1)
        func = self.b.size**-1 * logreg + self.regcoef / 2.0 * scipy.linalg.norm((self.last_xalphad_directional))**2
        return np.squeeze(func)

    def grad_directional(self, x, d, alpha):
        self.update_d_directional(d)
        self.update_xalphad_directional(x, d, alpha)
        logreg = np.dot(self.b * scipy.special.expit(-self.b * self.last_Axalphad_directional), self.last_Ad_directional)
        grad = -self.b.size**-1 * logreg + self.regcoef * np.dot(self.last_xalphad_directional, self.last_d_directional)
        return np.squeeze(grad)

    def hess_vec(self, x, d):
        self.update_x_directional(d)
        self.update_d_directional(d)

        sigma = scipy.special.expit(self.b * self.last_Ax_directional)
        logreg = self.matvec_ATx(sigma * (1 - sigma) * self.last_Ad_directional)
        return self.b.size**-1 * logreg + self.regcoef * d


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """

    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    def matmat_ATsA(s):
        if isinstance(A, scipy.sparse.spmatrix):
            return A.T.dot(scipy.sparse.diags(s)).dot(A)
        return np.dot(A.T, s[:, np.newaxis] * A)
    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """

    eps_e = eps * np.identity(x.size)
    eps_v = eps * v
    row_func = lambda e_i: func(x + eps_v + e_i) - func(x + e_i)

    E_i = np.apply_along_axis(row_func, 1, eps_e)
    return (E_i - func(x + eps_v) + func(x)) / eps**2
