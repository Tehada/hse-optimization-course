from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from datetime import datetime

import scipy
import scipy.sparse

import task1.optimization
import task1.oracles


class Timer:
    def __init__(self):
        self.start = datetime.now()

    def seconds(self):
        now = datetime.now()
        timedelta = now - self.start
        return timedelta.seconds + timedelta.microseconds * 1e-6


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is decreased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0).astype(np.float64)
    u_k = np.copy(u_0).astype(np.float64)
    t_k = t_0

    timer = Timer()
    converge = False
    n = x_k.size

    line_search_tool = task1.optimization.LineSearchTool(method='Armijo', c1=c1)
    ATA = None

    for num_iter in range(max_iter + 1):
        duality_gap = oracle.duality_gap(x_k) if lasso_duality_gap is not None else None

        if duality_gap is None:
            converge = True

        if trace:
            history['time'].append(timer.seconds())
            history['func'].append(np.copy(f_k))
            history['duality_gap'].append(np.copy(duality_gap))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display: print('step', history['time'][-1] if history else '')

        if duality_gap is not None and duality_gap <= tolerance:
            converge = True
            break

        if num_iter == max_iter: break

        if ATA is None:
            ATA = A.T.dot(A)

        class SubtaskOracle(task1.oracles.BaseSmoothOracle):
            def __init__(self, t):
                self.matvec_Ax = lambda x: A.dot(x)
                self.matvec_ATx = lambda x: A.T.dot(x)
                self.t = t

            def func(self, x):
                Ax_b = self.matvec_Ax(x[:n]) - b
                regression = 0.5 * np.dot(Ax_b, Ax_b)
                regularization = reg_coef * np.sum(x[n:])

                @np.vectorize
                def fixed_log(x):
                    return np.log(x) if x > 0 else np.inf

                return self.t * (regression + regularization) - np.sum(fixed_log(x[n:] + x[:n]) + fixed_log(x[n:] - x[:n]))

            def grad(self, x):
                regression = self.t * self.matvec_ATx(self.matvec_Ax(x[:n]) - b)
                regularization = np.full(n, self.t * reg_coef)
                left = 1.0 / (x[n:] + x[:n])
                right = 1.0 / (x[n:] - x[:n])

                return np.concatenate((regression - left + right, regularization - left - right))

            def hess(self, x):
                left = 1.0 / (x[n:] + x[:n])**2
                right = 1.0 / (x[n:] - x[:n])**2

                return np.bmat([
                    [self.t * ATA + np.diag(left + right), np.diag(left - right)],
                    [np.diag(left - right), np.diag(left + right)]
                ])

        newton_solution, newton_message, _ = task1.optimization.newton(
            SubtaskOracle(t_k),
            np.concatenate((x_k, u_k)),
            tolerance=tolerance_inner,
            max_iter=max_iter_inner,
            line_search_options=line_search_tool
        )

        #assert(newton_message is not 'newton_direction_error')

        x_k, u_k = newton_solution[:n], newton_solution[n:]
        t_k *= gamma

        print(x_k)

    return x_k, 'success' if converge else 'iterations_exceeded', history


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If avaliable, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0).astype(np.float64)

    timer = Timer()
    converge = False

    x_best, f_best = None, None

    for num_iter in range(max_iter + 1):
        f_k = oracle.func(x_k)

        duality_gap = oracle.duality_gap(x_k) if hasattr(oracle, 'duality_gap') else None

        if duality_gap is None:
            converge = True

        if f_best is None or f_best > f_k:
            x_best, f_best = np.copy(x_k), f_k

        if trace:
            history['time'].append(timer.seconds())
            history['func'].append(f_k)
            history['duality_gap'].append(duality_gap)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display: print('step', history['time'][-1] if history else '')

        if duality_gap is not None and duality_gap <= tolerance:
            converge = True
            break

        if num_iter == max_iter: break

        alpha_k = alpha_0 / np.sqrt(num_iter + 1)
        subgrad_k = oracle.subgrad(x_k)
        x_k -= alpha_k * subgrad_k / scipy.linalg.norm(subgrad_k)

    return x_best, 'success' if converge else 'iterations_exceeded', history


def proximal_gradient_descent(oracle, x_0, L_0=1, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Proximal gradient descent for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented
        for computing function value, its gradient and proximal mapping
        respectively.
        If avaliable, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0).astype(np.float64)
    f_k = oracle.func(x_k)

    timer = Timer()
    converge = False
    L_k = L_0
    last_nesterov_num_iterations = 0

    for num_iter in range(max_iter + 1):
        duality_gap = oracle.duality_gap(x_k) if hasattr(oracle, 'duality_gap') else None

        if duality_gap is None:
            converge = True


        if trace:
            history['time'].append(timer.seconds())
            history['func'].append(f_k)
            history['duality_gap'].append(duality_gap)
            history['nesterov_num_iterations'].append(last_nesterov_num_iterations)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display: print('step', history['time'][-1] if history else '')

        if duality_gap is not None and duality_gap <= tolerance:
            converge = True
            break

        if num_iter == max_iter: break

        _f_k = oracle._f.func(x_k)
        grad_k = oracle.grad(x_k)

        nesterov_converge = False
        last_nesterov_num_iterations = 0
        while not nesterov_converge:
            last_nesterov_num_iterations += 1

            def m(y, L):
                return _f_k + np.dot(grad_k, y - x_k) + L / 2.0 * np.dot(y - x_k, y - x_k) + oracle._h.func(y)

            alpha = 1.0 / L_k
            y = oracle.prox(x_k - alpha * grad_k, alpha)
            f_y = oracle.func(y)

            if f_y <= m(y, L_k):
                nesterov_converge = True
            else:
                L_k *= 2.0

        x_k, f_k = y, f_y
        L_k = max(L_0, L_k / 2.0)

    return x_k, 'success' if converge else 'iterations_exceeded', history
