from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from datetime import datetime
from numpy.linalg import LinAlgError

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


def newton(oracle, x_0, s, q, tolerance=1e-5, max_iter=100, theta=0.99,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = task1.optimization.get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(np.float64)

    timer = Timer()
    converge = False

    for num_iter in range(max_iter + 1):
        if np.isinf(x_k).any() or np.isnan(x_k).any():
            return x_k, 'computational_error', history

        f_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)

        if np.isinf(grad_k).any() or np.isnan(grad_k).any():
            return x_k, 'computational_error', history

        grad_norm_k = scipy.linalg.norm(grad_k)

        if trace:
            history['time'].append(timer.seconds())
            history['func'].append(np.copy(f_k))
            history['grad_norm'].append(np.copy(grad_norm_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display: print('step', history['time'][-1] if history else '')

        if num_iter == 0:
            eps_grad_norm_0 = np.sqrt(tolerance) * grad_norm_k
        if grad_norm_k <= eps_grad_norm_0:
            converge = True
            break

        if num_iter == max_iter: break

        hess_k = oracle.hess(x_k)
        if isinstance(hess_k, scipy.sparse.spmatrix):
            hess_k = hess_k.toarray()

        try:
            c, lower = scipy.linalg.cho_factor(hess_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        d_k = scipy.linalg.cho_solve((c, lower), -grad_k)

        q_d_k = q.dot(d_k)
        check = (s - q.dot(x_k)) / q_d_k
        check = check[q_d_k > 0.]

        alpha_0 = min(1.0, theta * np.min(check)) if check.size else 1.0
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, alpha_0)

        x_k += alpha_k * d_k

    return x_k, 'success' if converge else 'iterations_exceeded', history


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

    In the outer loop `t` is increased and we have a sequence of approximations
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

    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    timer = Timer()
    converge = False
    n = x_k.size

    class SubtaskOracle(task1.oracles.BaseSmoothOracle):
            def __init__(self, t):
                self.t = t

            def func(self, x):
                Ax_b = matvec_Ax(x[:n]) - b
                regression = 0.5 * np.dot(Ax_b, Ax_b)
                regularization = reg_coef * scipy.linalg.norm(x[n:], ord=1)

                @np.vectorize
                def fixed_log(x):
                    return np.log(x) if x > 0 else np.inf

                return self.t * (regression + regularization) - np.sum(fixed_log(x[n:] + x[:n]) + fixed_log(x[n:] - x[:n]))

            def grad(self, x):
                regression = matvec_ATx(matvec_Ax(x[:n]) - b)
                regularization = np.full(n, reg_coef)

                left = 1.0 / (self.t * x[n:] + self.t * x[:n])
                right = 1.0 / (self.t * x[n:] - self.t * x[:n])

                return self.t * np.concatenate((regression - left + right, regularization - left - right))

            def hess(self, x):
                left = 1.0 / (x[n:] + x[:n])**2
                right = 1.0 / (x[n:] - x[:n])**2

                return np.bmat([
                    [self.t * ATA + np.diag(left + right), np.diag(left - right)],
                    [np.diag(left - right), np.diag(left + right)]
                ])

    ATA = A.T.dot(A)

    q = np.bmat([
        [np.eye(n), -1 * np.eye(n)],
        [-1 * np.eye(n), -1 * np.eye(n)],
    ])

    def eval_duality_gap(x):
        if lasso_duality_gap is None:
            return None
        Ax_b = matvec_Ax(x) - b
        ATAx_b = matvec_ATx(Ax_b)
        return lasso_duality_gap(x, Ax_b, ATAx_b, b, reg_coef)

    for num_iter in range(max_iter + 1):
        duality_gap = eval_duality_gap(x_k)

        if duality_gap is None:
            converge = True

        if trace:
            history['time'].append(timer.seconds())
            history['func'].append(0.5 * matvec_Ax(x_k) - b + reg_coef * scipy.linalg.norm(x_k, ord=1))
            history['duality_gap'].append(duality_gap)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display: print('step', history['time'][-1] if history else '')

        if duality_gap is not None and duality_gap <= tolerance:
            converge = True
            break

        if num_iter == max_iter: break

        newton_solution, newton_message, _ = newton(
            SubtaskOracle(t_k),
            np.concatenate((x_k, u_k)),
            np.zeros(2 * n), q, theta=0.99,
            tolerance=tolerance_inner,
            max_iter=max_iter_inner,
            line_search_options={'method': 'Armijo', 'c1': c1}
        )

        # assert newton_message is not 'newton_direction_error'

        x_k, u_k = newton_solution[:n], newton_solution[n:]

        t_k *= gamma

    return (x_k, u_k), 'success' if converge else 'iterations_exceeded', history


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
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
        If available, .duality_gap() method is used for estimating f_k - f*.
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

            def m(y, L, _h_y):
                if _h_y is None: _h_y = oracle._h.func(y)
                return _f_k + np.dot(grad_k, y - x_k) + L / 2.0 * np.dot(y - x_k, y - x_k) + _h_y

            alpha = 1.0 / L_k
            y = oracle.prox(x_k - alpha * grad_k, alpha)
            _f_y = oracle._f.func(y)
            _h_y = oracle._h.func(y)
            f_y = _f_y + _h_y

            if f_y <= m(y, L_k, _h_y):
                nesterov_converge = True
            else:
                L_k *= 2.0

        x_k, f_k = y, f_y
        L_k = max(L_0, L_k / 2.0)

    return x_k, 'success' if converge else 'iterations_exceeded', history
