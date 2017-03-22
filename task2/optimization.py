import numpy as np
import scipy
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from datetime import datetime
from utils import get_line_search_tool


class Timer:
    def __init__(self):
        self.start = datetime.now()

    def seconds(self):
        now = datetime.now()
        timedelta = now - self.start
        return timedelta.seconds + timedelta.microseconds * 1e-6


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

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
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None

    x_k = np.copy(x_0).astype(np.float64)
    g_k = matvec(x_k) - b
    d_k = -g_k
    eps_b_norm = tolerance * scipy.linalg.norm(b)

    timer = Timer()
    converge = False

    if max_iter is None:
        max_iter = x_k.size

    for num_iter in range(max_iter + 1):
        g_k_norm = scipy.linalg.norm(g_k)

        if trace:
            history['time'].append(timer.seconds())
            history['residual_norm'].append(np.copy(g_k_norm))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display: print('step', history['time'][-1] if history else '')

        if g_k_norm <= eps_b_norm:
            converge = True
            break
        if num_iter == max_iter: break

        A_d_k = matvec(d_k)
        g_k_sq = np.dot(g_k, g_k)
        d_coef = g_k_sq / np.dot(d_k, A_d_k)

        x_k += d_coef * d_k
        g_k += d_coef * A_d_k
        d_k = -g_k + np.dot(g_k, g_k) / g_k_sq * d_k

    return x_k, 'success' if converge else 'iterations_exceeded', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(np.float64)

    timer = Timer()
    converge = False
    alpha_k = None

    s_trace, y_trace = deque(), deque()
    grad_k = oracle.grad(x_k)

    for num_iter in range(max_iter + 1):
        # if np.isinf(x_k).any() or np.isnan(x_k).any():
        #     return x_k, 'computational_error', history

        f_k = oracle.func(x_k)

        # if np.isinf(grad_k).any() or np.isnan(grad_k).any():
        #     return x_k, 'computational_error', history

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


        def lbfgs_direction(grad, s_trace, y_trace):
            d = -grad

            if not s_trace:
                return d

            mus = []
            for s, y in zip(reversed(s_trace), reversed(y_trace)):
                mu = np.dot(s, d) / np.dot(s, y)
                mus.append(mu)
                d -= mu * y

            d *= np.dot(s_trace[-1], y_trace[-1]) / np.dot(y_trace[-1], y_trace[-1])

            for s, y, mu in zip(s_trace, y_trace, reversed(mus)):
                beta = np.dot(y, d) / np.dot(s, y)
                d += (mu - beta) * s

            return d


        d_k = lbfgs_direction(grad_k, s_trace, y_trace)
        alpha_k = line_search_tool.line_search(
            oracle, x_k, d_k,
            2.0 * alpha_k if alpha_k is not None else None
        )
        x_k += alpha_k * d_k
        last_grad_k = np.copy(grad_k)
        grad_k = oracle.grad(x_k)

        if memory_size > 0:
            if len(s_trace) == memory_size:
                s_trace.popleft()
                y_trace.popleft()
            s_trace.append(alpha_k * d_k)
            y_trace.append(grad_k - last_grad_k)


    return x_k, 'success' if converge else 'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(np.float64)

    timer = Timer()
    converge = False
    alpha_k = None

    for num_iter in range(max_iter + 1):
        # if np.isinf(x_k).any() or np.isnan(x_k).any():
        #     return x_k, 'computational_error', history

        f_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)

        # if np.isinf(grad_k).any() or np.isnan(grad_k).any():
        #     return x_k, 'computational_error', history

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

        eta = min(0.5, np.sqrt(grad_norm_k))
        conjugate_gradient_converge = False

        while not conjugate_gradient_converge:
            d_k, _, _ = conjugate_gradients(lambda d: oracle.hess_vec(x_k, d), -grad_k, -grad_k, tolerance=eta)
            eta /= 10
            conjugate_gradient_converge = np.dot(d_k, grad_k) < 0

        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, 1.0)
        x_k += alpha_k * d_k

    return x_k, 'success' if converge else 'iterations_exceeded', history
