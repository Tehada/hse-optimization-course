import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict


class Timer:
    def __init__(self):
        self.start = datetime.now()

    def seconds(self):
        now = datetime.now()
        timedelta = now - self.start
        return timedelta.seconds + timedelta.microseconds * 1e-6


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """

        if self._method == 'Constant':
            return self.c

        phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
        derphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)
        phi0 = phi(0.0)
        derphi0 = derphi(0.0)

        if self._method == 'Wolfe':
            alpha_wolfe = scipy.optimize.linesearch.scalar_search_wolfe2(
                phi, derphi, phi0=phi0, derphi0=derphi0, c1=self.c1, c2=self.c2
            )
            if alpha_wolfe[0] is not None:
                return alpha_wolfe[0]

        if previous_alpha is None:
            alpha_k = np.copy(self.alpha_0)
        else:
            alpha_k = np.copy(previous_alpha)

        while phi(alpha_k) > phi0 + self.c1 * alpha_k * derphi0:
            alpha_k = np.divide(alpha_k, 2.0)

        return alpha_k


def get_line_search_tool(line_search_options=None):
      if line_search_options:
         if type(line_search_options) is LineSearchTool:
             return line_search_options
         else:
             return LineSearchTool.from_dict(line_search_options)
      else:
          return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
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
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(np.float64)

    timer = Timer()
    converge = False
    alpha_k = None

    for num_iter in range(max_iter + 1):
        if np.isinf(x_k).any() or np.isnan(x_k).any():
            return x_k, 'computational_error', history

        f_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)

        if np.isinf(f_k) or np.isnan(f_k) or np.isinf(grad_k).any() or np.isnan(grad_k).any():
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

        d_k = -grad_k
        alpha_k = line_search_tool.line_search(
            oracle, x_k, d_k,
            2.0 * alpha_k if alpha_k is not None else None
        )
        x_k += alpha_k * d_k

    return x_k, 'success' if converge else 'iterations_exceeded', history



def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.
    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string`
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(np.float64)

    timer = Timer()
    converge = False
    alpha_k = None

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
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, 1.0)
        x_k += alpha_k * d_k

    return x_k, 'success' if converge else 'iterations_exceeded', history
