from math import e
import warnings
import math
import csv
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import tailer
import io

from zmq import strerror

from ..optimize import Optimizer
from .spline_inner_problem import SplineInnerProblem
from .parameter import InnerParameter
from .problem import InnerProblem
from .solver import InnerSolver

REDUCED = "reduced"
STANDARD = "standard"
MAXMIN = "max-min"
MAX = "max"


class SplineInnerSolver(InnerSolver):
    """
    Solver of the inner subproblem.

    Options
    -------
    spline_ratio:
        Wanted ratio of the number of spline parameters and the number 
        of measurements. Can be any positive float. Default value is 1/2.
    inner_optimizer:
        Optimizer of the inner problem. Default is SLSQP.
    minimal_difference:
        If True then the method will constrain minimal spline parameter
        difference. Otherwise there will be no such constrain.
    """

    def __init__(self, optimizer: Optimizer = None, options: Dict = None):

        self.optimizer = optimizer
        self.options = options
        if self.options is None:
            self.options = SplineInnerSolver.get_default_options()
        else:
            if not type(self.options['spline_ratio']) == float:
                raise ValueError("Spline ratio must be a positive float.")
            if self.options['spline_ratio'] <= 0:
                raise ValueError("Spline ratio must be a positive float.")
            if self.options['inner_optimizer'] not in ['SLSQP', 'LS', 'fides']:
                raise ValueError("Chosen Inner optimizer {inner_optimizer} is not implemented. Choose from SLSQP, LS or fides")
            if self.options['minimal_difference'] not in [True, False]:
                raise ValueError('Minimal difference must be a boolean value.')

    def solve(
        self,
        problem: InnerProblem,
        sim: List[np.ndarray],
        sigma: List[np.ndarray],
    ) -> list:
        """
        Get results for every group (inner optimization problem)

        Parameters
        ----------
        problem:
            InnerProblem from pyPESTO hierarchical
        sim:
            Simulations from AMICI
        sigma:
            List of sigmas (not needed for this approach)
        """
        inner_optimization_results = []
        simulation_indices = problem.simulation_indices
        inner_optimizer = self.options['inner_optimizer']

        for gr in problem.get_groups_for_xs(InnerParameter.OPTIMALSCALING):
            sigma_for_group = get_sigma_for_group(gr, sigma, simulation_indices)
            quantitative_data = problem.get_quantitative_data_for_group(gr)
            quantitative_data = quantitative_data.sort_values(
                by=["simulationConditionId", "time"]
            )

            inner_optimization_results_per_group = optimize_spline(
                gr,
                sim,
                quantitative_data,
                simulation_indices,
                self.options,
                sigma_for_group,
                inner_optimizer,
            )
            inner_optimization_results.append(inner_optimization_results_per_group)
        return inner_optimization_results

    @staticmethod
    def calculate_obj_function(x_inner_opt: list):
        """
        Calculate the inner objective function from a list of inner
        optimization results returned from optimize_spline

        Parameters
        ----------
        x_inner_opt:
            List of optimization results
        """

        obj = np.sum([x_inner_opt[idx]["fun"] for idx in range(len(x_inner_opt))])
        return obj

    @staticmethod
    def ls_calculate_obj_function(x_inner_opt: list):
        """
        Calculate the inner objective function from a list of inner
        optimization results returned from optimize_spline
        using the least squares inner optimizer

        Parameters
        ----------
        x_inner_opt:
            List of optimization results
        """
        obj = np.sum([x_inner_opt[idx]["fun"][0] for idx in range(len(x_inner_opt))])
        return obj

    def calculate_gradients_reformulated(
        self,
        problem: SplineInnerProblem,
        x_inner_opt,
        sim,
        sy,
        parameter_mapping,
        par_opt_ids,
        amici_model,
        snllh,
        sigma_full,
    ):
        """
        Calculate the gradient of the objective function with respect to
        the outer dynaical parameters. 

        Parameters
        ----------
        problem:
            SplineInnerProblem
        """
        simulation_indices = problem.simulation_indices
        condition_map_sim_var = parameter_mapping[0].map_sim_var
        par_sim_idx = -1
        # TODO: Doesn't work with condition specific parameters
        for par_sim, par_opt in condition_map_sim_var.items():
            if not isinstance(par_opt, str):
                continue
            if par_opt.startswith("optimalScaling_"):
                continue
            par_sim_idx += 1
            inner_par_idx = 0
            par_opt_idx = par_opt_ids.index(par_opt)

            grad = 0.0
            for idx, gr in enumerate(
                problem.get_groups_for_xs(InnerParameter.OPTIMALSCALING)
            ):
                #the reformulated spline parameters
                s = np.asarray(x_inner_opt[inner_par_idx]["x"])

                sim_all = get_sim_all(gr, sim, simulation_indices)
                sigma = get_sigma_for_group(gr, sigma_full, simulation_indices)

                inner_par_idx += 1

                sy_all = get_sy_all(
                    gr, sy, par_sim_idx, simulation_indices
                )
                quantitative_data = problem.get_quantitative_data_for_group(gr)
                quantitative_data = quantitative_data.sort_values(
                    by=["simulationConditionId", "time"]
                )

                measurements = quantitative_data.measurement.values

                N, delta_c, c, n = rescale_spline_bases(
                    sim_all, self.options["spline_ratio"]
                )
                delta_c_dot, c_dot = get_spline_bases_gradient(
                    sim_all, sy_all, self.options["spline_ratio"]
                )

                C = np.diag(-np.ones(N))
                mu = get_gradient_reformulated(
                    s, sim_all, measurements, sigma, N, delta_c, c, n
                )

                # Correcting for small errors in optimization/calculations
                for i in range(len(mu)):
                    if abs(mu[i]) < 1e-5:
                        mu[i] = 0

                # Calculate df_ds term only if mu is not all 0
                if np.any(mu):
                    s_dot = get_ds_dtheta(
                        sim_all,
                        sy_all,
                        measurements,
                        s,
                        C,
                        mu,
                        sigma,
                        N,
                        delta_c,
                        delta_c_dot,
                        c,
                        c_dot,
                        n,
                        self.options["minimal_difference"]
                    )
                    df_ds = mu
                    grad += df_ds.dot(s_dot)

                # Let's calculate the df_dyk term now:
                df_dyk = 0

                for y_k, z_k, y_dot_k, sigma_k, n_k in zip(
                    sim_all, measurements, sy_all, sigma, n
                ):

                    i = n_k - 1
                    sum_s = 0
                    for j in range(i):
                        sum_s += s[j]
                    if i > 0 and i < N:
                        df_dyk += (
                            (1 / sigma_k ** 2)
                            * ((y_k - c[i - 1]) * s[i] / delta_c + sum_s - z_k)
                            * s[i]
                            * (
                                (y_dot_k - c_dot[i - 1]) * delta_c
                                - (y_k - c[i - 1]) * delta_c_dot
                            )
                            / delta_c ** 2
                        )

                grad += df_dyk
            snllh[par_opt_idx] = grad

        return snllh

    @staticmethod
    def get_default_options() -> Dict:
        """
        Return default options for solving the inner problem,
        if no options provided
        """
        options={
            "spline_ratio": 1/2,
            "inner_optimizer": 'SLSQP',
            "minimal_difference": True,
        }
        return options


def optimize_spline(
    gr: float,
    sim: List[np.ndarray],
    quantitative_data: pd.DataFrame,
    simulation_indices: List,
    options: Dict,
    sigma: np.ndarray,
    inner_optimizer: str,
):
    """Run optimization for inner problem"""

    from scipy.optimize import minimize, least_squares
    import fides

    sim_all = get_sim_all(gr, sim, simulation_indices)

    measurements = quantitative_data.measurement.values

    N, delta_c, c, n = rescale_spline_bases(sim_all, options["spline_ratio"])

    min_meas, max_meas, min_diff = get_min_max_min_diff(measurements, N, options["minimal_difference"])
    
    inner_options = get_spline_inner_options(N, min_meas, max_meas, min_diff)

    def obj_surr_reformulated(x):
        return obj_spline_reformulated(
            x, sim_all, measurements, sigma, N, delta_c, c, n
        )

    def obj_jac_reformulated(x):
        return get_gradient_reformulated(
            x, sim_all, measurements, sigma, N, delta_c, c, n
        )
    

    if inner_optimizer == 'SLSQP':


        results = minimize(obj_surr_reformulated, jac=obj_jac_reformulated, **inner_options)
        results["x"][0] = results["x"][0].clip(min=0)
        results["x"][1:] = results["x"][1:].clip(min=min_diff)
    
    elif inner_optimizer == 'LS':

        results = least_squares(obj_surr_reformulated, inner_options['x0'], jac=obj_jac_reformulated, bounds=(0,np.inf))

    elif inner_optimizer == 'fides':


        def obj_hess_reformulated(x):
            return get_Hessian_reformulated(x, sim_all, measurements, sigma, N, delta_c, c, n)

        def obj_fides(x):
            return obj_spline_reformulated(x, sim_all, measurements, sigma, N, delta_c, c, n), \
                get_gradient_reformulated(x, sim_all, measurements, sigma, N, delta_c, c, n), \
                get_Hessian_reformulated(x, sim_all, measurements, sigma, N, delta_c, c, n)

        lower_bounds = np.full(N, min_diff)
        upper_bounds = np.full(N, np.inf)
        lower_bounds[0]=0
        opt_fides = fides.Optimizer(obj_fides, ub = upper_bounds, lb=lower_bounds)

        results = opt_fides.minimize(inner_options['x0'])
    
    results["x"][0] = results["x"][0].clip(min=0)
    results["x"][1:] = results["x"][1:].clip(min=min_diff)

    return results


def obj_spline(
    xi: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Objective function for non-reformulated problem 
    with xi as spline parameters"""
    obj = 0
    for y_k, z_k, sigma_k in zip(sim_all, measurements, sigma):
        n = math.ceil((y_k - c[0]) / delta_c) + 1
        if n == 0:
            n = 1
        if n > N:
            n = N + 1
        i = n - 1
        if n < N + 1 and n > 1:
            obj += (1 / sigma_k ** 2) * (
                z_k
                - (y_k - c[0] - delta_c * (n - 2))
                * (xi[i] - xi[i - 1])
                / delta_c
                - xi[i - 1]
            ) ** 2
        elif n == N + 1:
            obj += (1 / sigma_k ** 2) * (z_k - xi[i - 1]) ** 2
        elif n == 1:
            obj += (1 / sigma_k ** 2) * (z_k - y_k * xi[i] / c[0]) ** 2
        obj += math.log(2 * math.pi * sigma_k ** 2)
    obj = obj / 2
    return obj


def obj_spline_reformulated(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """ Objective function for problem reformulation 
    with s as parmaeter splines."""
    obj = 0

    for y_k, z_k, sigma_k, n_k in zip(sim_all, measurements, sigma, n):
        i = n_k - 1
        sum_s = 0
        for j in range(i):
            sum_s += s[j]
        if i == 0:
            obj += (1 / sigma_k ** 2) * (z_k - s[i]) ** 2
        elif i == N:
            obj += (1 / sigma_k ** 2) * (z_k - sum_s) ** 2
        else:
            obj += (1 / sigma_k ** 2) * (
                z_k - (y_k - c[i - 1]) * s[i] / delta_c - sum_s
            ) ** 2
    obj = obj / 2
    return obj


def get_spline_inner_options(N: int, min_meas: float, max_meas: float, min_diff: float) -> Dict:

    """Return options for optimization"""
    # TODO not implemented, start at last optimal spline parameters.
    # try:
    #     with open('/home/zebo/Desktop/numerical_spline_xi.csv') as file:
    #         last_lines = tailer.tail(file, 4)
    #     last_inner_runs = pd.read_csv(io.StringIO('\n'.join(last_lines)),
    #                             names=['gr', 'N', 'delta_c', 'c_1', 'inversions', 'xi_1', 'xi_2', 'xi_3', 'xi_4', 'xi_5', 'xi_6', 'xi_7', 'xi_8', 'xi_9', 'xi_10', 'xi_11', 'xi_12', 'xi_13', 'Sim_1', 'Sim_2', 'Sim_3', 'Sim_4', 'Sim_5', 'Sim_6', 'Sim_7', 'Sim_8', 'Sim_9', 'Sim_10', 'Sim_11'])

    #     last_inner_runs = last_inner_runs[last_inner_runs['gr']==gr]

    #     x0 = np.zeros(int(N))
    #     for k in range(1, int(N)+1):
    #         x0[k-1] = last_inner_runs.iloc[[-1]]['xi_' + str(k)].values[0]
    # except:
    range_all = max_meas - min_meas

    constraint_min_diff = np.full(N, min_diff)
    constraint_min_diff[0] = 0

    x0 = np.full(
        N,
        (max_meas + 0.3 * range_all - np.max([min_meas - 0.3 * range_all, 0]))
        / (N - 1),
    )
    x0[0] = np.max([min_meas - 0.3 * range_all, 0])

    inner_options = {
        "x0": x0,
        "method": "SLSQP",
        "options": {"maxiter": 2000, "ftol": 1e-10, "disp": None},
        "constraints": {"type": "ineq", "fun": lambda x: x - constraint_min_diff},
    }

    return inner_options


def get_min_max_min_diff(measurements: np.ndarray, N: int, minimal_difference: bool):
    """
    Return minimal measurement, maximal measurement
    and minimal parameter difference for spline parameters
    """

    min_all = max_all = measurements[0]
    for m in measurements:
        if m > max_all:
            max_all = m
        if m < min_all:
            min_all = m
    range_all = max_all - min_all
    if minimal_difference:
        min_diff = range_all / (2 * N)
    else:
        min_diff = 0
    return min_all, max_all, min_diff


def spline_get_gradient(
    sim_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """ Gradient of the objective function with respect 
    to the spline parameters for the non-reformulated problem.
    Returns the linearized form of the gradient:
    
    gradient = lhs_matrix * xi - rhs

    """
    lhs_matrix = np.zeros((N, N))
    rhs = np.zeros(N)

    for y_k, z_k, sigma_k in zip(sim_all, measurements, sigma):
        n = math.ceil((y_k - c[0]) / delta_c) + 1
        if n == 0:
            n = 1
        if n > N:
            n = N + 1
        i = n - 1  # just the iterator to go over the Jacobian matrix
        if n < N + 1:
            if n > 1:
                lhs_matrix[i][i - 1] += (
                    (1 / sigma_k ** 2)
                    * (y_k - c[0] - (n - 2) * delta_c)
                    * (c[0] + (n - 1) * delta_c - y_k)
                )
            lhs_matrix[i][i] += (1 / sigma_k ** 2) * (y_k - c[0] - (n - 2) * delta_c) ** 2
            rhs[i] += (
                (1 / sigma_k ** 2) * z_k * (y_k - c[0] - (n - 2) * delta_c) * delta_c
            )
        if n > 1 and n < N + 1:
            lhs_matrix[i - 1][i - 1] += (1 / sigma_k ** 2) * (
                c[0] + (n - 1) * delta_c - y_k
            ) ** 2
            if n < N + 1:
                lhs_matrix[i - 1][i] += (
                    (1 / sigma_k ** 2)
                    * (y_k - c[0] - (n - 2) * delta_c)
                    * (c[0] + (n - 1) * delta_c - y_k)
                )
            rhs[i - 1] += (
                (1 / sigma_k ** 2) * z_k * (c[0] + (n - 1) * delta_c - y_k) * delta_c
            )
        if n == N + 1:
            lhs_matrix[i - 1][i - 1] += (1 / sigma_k ** 2) * delta_c ** 2
            rhs[i - 1] += (1 / sigma_k ** 2) * z_k * delta_c ** 2
    lhs_matrix = np.divide(lhs_matrix, delta_c ** 2)
    rhs = np.divide(rhs, delta_c ** 2)
    return lhs_matrix, rhs


def get_gradient_reformulated(
    optimal_s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """ Gradient of the objective function with respect 
    to the spline parameters for the reformulated inner problem """

    Gradient = np.zeros(N)

    for y_k, z_k, sigma_k, n_k in zip(sim_all, measurements, sigma, n):

        weight_k = 1 / sigma_k ** 2
        sum_s = 0
        i = n_k - 1  # just the iterator to go over the Jacobian array
        for j in range(i):
            sum_s += optimal_s[j]
        if i == 0:
            Gradient[i] += weight_k * (optimal_s[i] - z_k)
        elif i == N:
            for j in range(i):
                Gradient[j] += weight_k * (sum_s - z_k)
        else:
            Gradient[i] += (
                weight_k
                * ((y_k - c[i - 1]) * optimal_s[i] / delta_c + sum_s - z_k)
                * (y_k - c[i - 1])
                / delta_c
            )
            Gradient[:i] += np.full(
                i, weight_k * ((y_k - c[i - 1]) * optimal_s[i] / delta_c + sum_s - z_k)
            )
    return Gradient


def get_Hessian_reformulated(
    optimal_s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """ Hessian of the objective function with respect 
    to the spline parameters for the reformulated inner problem """

    Hessian = np.zeros((N, N))

    for y_k, z_k, sigma_k, n_k in zip(sim_all, measurements, sigma, n):
        sum_s = 0
        i = n_k - 1  # just the iterator to go over the Hessian matrix
        for j in range(i):
            sum_s += optimal_s[j]
        # ALSO MAKE THE LAST MONOTONICITY STEP
        Hessian[i][i] += (1 / sigma_k ** 2) * ((y_k - c[i - 1]) / delta_c) ** 2
        for j in range(i):
            Hessian[i][j] += (1 / sigma_k ** 2) * ((y_k - c[i - 1]) / delta_c)
            Hessian[j][i] += (1 / sigma_k ** 2) * ((y_k - c[i - 1]) / delta_c)
            for h in range(i):
                Hessian[j][h] += 1 / sigma_k ** 2
    # print(np.linalg.eig(Hessian))
    return Hessian


def get_dxi_dtheta(gr, sim_all, sy_all, measurements, xi, C, mu, sigma, N, delta_c, delta_c_dot, c, c_dot, n):
    """ 
    Calculates the derivative of spline parameters xi with respect to the 
    dynamical parameter theta. 
    Firstly, we calculate the derivative of the first two equations of 
    the necessary optimality conditions of the optimization problem with
    inequality constraints. Then we solve the linear system to obtain the
    derivatives.
    The derivative of the gradient with respect to the dynamical parameter
    theta is calculated in the linearlized form with respect to dxi_dtheta:

    d_{theta} (gradient_{xi}) = 
     = gradient_derivative_lhs * dxi_dtheta - gradient_derivative_rhs
    
    In the thesis, we noted gradient_derivative_lhs and gradient_derivative_rhs
    as M_i and R_i, respectively.
    """

    gradient_derivative_lhs = np.zeros((N, N))
    gradient_derivative_rhs = np.zeros(2 * N)

    for y_k, z_k, y_dot_k, sigma_k in zip(sim_all, measurements, sy_all, sigma):
        n = math.ceil((y_k - c[0]) / delta_c) + 1
        if n == 0:
            n = 1
        if n > N:
            n = N + 1
        i = n - 1  # just the iterator to go over the Jacobian matrix
        # calculate the Jacobian derivative:
        if n == N + 1:
            # print("U SEDAM SAAAAM LEEEEL")
            gradient_derivative_lhs[i - 1][i - 1] += (1 / sigma_k ** 2) * delta_c ** 2
            # rhs[i-1] += -2*w_dot*( xi[i-1] - z_k)* delta_c**2 NEED TO ADD HERE IF SIGMA IS OPTIMIZED
        else:
            if n < N + 1:
                if n > 1:
                    gradient_derivative_lhs[i][i - 1] += (
                        (1 / sigma_k ** 2)
                        * (y_k - c[0] - (n - 2) * delta_c)
                        * (c[0] + (n - 1) * delta_c - y_k)
                    )
                gradient_derivative_lhs[i][i] += (1 / sigma_k ** 2) * (
                    y_k - c[0] - (n - 2) * delta_c
                ) ** 2
                if n > 1:
                    gradient_derivative_rhs[i] += -(1 / sigma_k ** 2) * xi[i - 1] * y_dot_k * (
                        2 * c[0] + (2 * n - 3) * delta_c - 2 * y_k
                    ) - (1 / sigma_k ** 2) * xi[i - 1] * (
                        y_k - c[0] - (n - 2) * delta_c
                    ) * (
                        c[0] + (n - 1) * delta_c - y_k
                    )
                gradient_derivative_rhs[i] += (
                    -(1 / sigma_k ** 2)
                    * xi[i]
                    * 2
                    * y_dot_k
                    * (y_k - c[0] - (n - 2) * delta_c)
                )
                gradient_derivative_rhs[i] += (1 / sigma_k ** 2) * z_k * delta_c * y_dot_k
            if n > 1:
                gradient_derivative_lhs[i - 1][i - 1] += (1 / sigma_k ** 2) * (
                    c[0] + (n - 1) * delta_c - y_k
                ) ** 2
                if n < N + 1:
                    gradient_derivative_lhs[i - 1][i] += (
                        (1 / sigma_k ** 2)
                        * (y_k - c[0] - (n - 2) * delta_c)
                        * (c[0] + (n - 1) * delta_c - y_k)
                    )
                if n < N + 1:
                    gradient_derivative_rhs[i - 1] += (
                        -(1 / sigma_k ** 2)
                        * xi[i]
                        * y_dot_k
                        * (2 * c[0] + (2 * n - 3) * delta_c - 2 * y_k)
                    )
                gradient_derivative_rhs[i - 1] += (
                    -(1 / sigma_k ** 2)
                    * xi[i - 1]
                    * 2
                    * y_dot_k
                    * (y_k - c[0] - (n - 1) * delta_c)
                )
                gradient_derivative_rhs[i - 1] += -(1 / sigma_k ** 2) * z_k * delta_c * y_dot_k

    gradient_derivative_lhs = np.divide(gradient_derivative_lhs, ((delta_c) ** 2))
    gradient_derivative_rhs = np.divide(gradient_derivative_rhs, ((delta_c) ** 2))

    if np.all((mu == 0)):
        from scipy import linalg

        gradient_derivative_rhs = gradient_derivative_rhs[:N]
        lhs = gradient_derivative_lhs

        dxi_dtheta = linalg.lstsq(lhs, gradient_derivative_rhs)
        return dxi_dtheta[0]
    else:
        from scipy.sparse import linalg, csc_matrix

        lhs = np.block(
            [
                [gradient_derivative_lhs, C.transpose()],
                [(mu * C.transpose()).transpose(), np.diag(C.dot(xi))],
            ]
        )

        lhs_sp = csc_matrix(lhs)

        dxi_dtheta = linalg.spsolve(lhs_sp, gradient_derivative_rhs)
        return dxi_dtheta[:N]


def get_ds_dtheta(
    sim_all, sy_all, measurements, s, C, mu, sigma, N, delta_c, delta_c_dot, c, c_dot, n, minimal_difference
):
    """Calculates the derivative of reformulated spline parameters s with respect to the 
    dynamical parameter theta. Look at 'get_dxi_dtheta()' for details."""

    Jacobian_derivative = np.zeros((N, N))
    rhs = np.zeros(2 * N)

    min_meas, max_meas, min_diff = get_min_max_min_diff(measurements, N, minimal_difference)

    for y_k, z_k, y_dot_k, sigma_k, n_k in zip(sim_all, measurements, sy_all, sigma, n):

        i = n_k - 1  # just the iterator to go over the Jacobian matrix
        weight_k = 1 / sigma_k ** 2
        sum_s = 0
        for j in range(i):
            sum_s += s[j]

        # calculate the Jacobian derivative:
        if i == 0:
            Jacobian_derivative[i][i] += weight_k
        elif i == N:
            Jacobian_derivative = Jacobian_derivative + np.full((N, N), weight_k)

        else:
            Jacobian_derivative[i][i] += weight_k * (y_k - c[i - 1]) ** 2 / delta_c ** 2
            rhs[i] += (
                weight_k
                * (2 * (y_k - c[i - 1]) / delta_c * s[i] + sum_s - z_k)
                * ((y_dot_k - c_dot[i - 1]) * delta_c - (y_k - c[i - 1]) * delta_c_dot)
                / delta_c ** 2
            )
            if i > 0:
                Jacobian_derivative[i, :i] += np.full(
                    i, weight_k * (y_k - c[i - 1]) / delta_c
                )
                Jacobian_derivative[:i, i] += np.full(
                    i, weight_k * (y_k - c[i - 1]) / delta_c
                )
                rhs[:i] += np.full(
                    i,
                    weight_k
                    * (
                        (y_dot_k - c_dot[i - 1]) * delta_c
                        - (y_k - c[i - 1]) * delta_c_dot
                    )
                    * s[i]
                    / delta_c ** 2,
                )
                Jacobian_derivative[:i, :i] += np.full((i, i), weight_k)

    from scipy import linalg

    constraint_min_diff = np.diag(np.full(N, min_diff))
    constraint_min_diff[0][0] = 0
    lhs = np.block(
        [[Jacobian_derivative, C], [-np.diag(mu), constraint_min_diff - np.diag(s)]]
    )
    ds_dtheta = linalg.lstsq(lhs, rhs, lapack_driver="gelsy")

    return ds_dtheta[0][:N]


def get_mu(Jacobian, rhs, xi, C):
    from scipy import linalg
    from scipy.optimize import least_squares, minimize

    """
    Calculate the Langrange multipliers mu for the non-reformulated problem.
    For the reformulated problem, they equal the gradient.
    """

    C_transpose = C.transpose()

    rhs_mu = -(Jacobian.dot(xi) - rhs)

    def mu_obj(x):
        return (C_transpose.dot(x) - rhs_mu).sum()

    constraints = [
        {"type": "ineq", "fun": lambda x: x},
        {
            "type": "ineq",
            "fun": lambda x: np.full(len(xi), 1e-6) - np.multiply(x, C.dot(xi)),
        },
    ]

    inner_options = {
        "x0": np.zeros(len(xi)),
        "method": "SLSQP",
        "options": {"maxiter": 2000, "ftol": 1e-10, "disp": True},
        "constraints": constraints,
    }

    mu_slsqp = minimize(mu_obj, **inner_options)
    print("mu_slsqp: ", mu_slsqp["x"])
    print("residue: ", (C_transpose.dot(mu_slsqp["x"]) - rhs_mu).sum())

    return mu_slsqp["x"]

def get_sim_all(
    gr, sim: List[np.ndarray], simulation_indices: List
) -> list:
    """ "Get list of all simulations for a group"""

    gr = int(gr) - 1
    sim_length = 0
    for condition_indx in range(len(simulation_indices[gr])):
        sim_length += len(simulation_indices[gr][condition_indx])
    sim_all = -np.ones((sim_length))

    current_indx = 0
    for condition_indx in range(len(simulation_indices[gr])):
        for time_indx in simulation_indices[gr][condition_indx]:
            sim_all[current_indx] = sim[condition_indx][time_indx][gr]
            current_indx += 1
    return sim_all


def get_sy_all(gr, sy, par_idx, simulation_indices):
    """ Get list of all sensitivities for a group"""
    gr = int(gr) - 1
    sim_length = 0
    for condition_indx in range(len(simulation_indices[gr])):
        sim_length += len(simulation_indices[gr][condition_indx])
    sy_all = -np.ones((sim_length))
    i = 0
    current_indx = 0
    for condition_indx in range(len(simulation_indices[gr])):
        for time_indx in simulation_indices[gr][condition_indx]:
            sy_all[current_indx] = sy[condition_indx][time_indx][par_idx][gr]
            current_indx += 1
    return sy_all

def rescale_spline_bases(sim_all, r):
    """ Rescale the spline bases """
    K = len(sim_all)

    min_all = max_all = sim_all[0]
    max_idx = min_idx = 0

    for idx in range(len(sim_all)):
        if sim_all[idx] > max_all:
            max_all = sim_all[idx]
            max_idx = idx
        if sim_all[idx] < min_all:
            min_all = sim_all[idx]
            min_idx = idx

    N = math.ceil(r * K)
    n = np.ones(K)

    # In case the simulation are very close to each other
    # or even collapse into a single point (e.g. steady-state)
    if max_all - min_all < 1e-6:
        average_value = (max_all + min_all) / 2
        if average_value < 5e-7:
            delta_c = 1e-6 / (N - 1)
            c = np.linspace(0, 1e-6, N)
        else:
            delta_c = 1e-6 / (N - 1)
            c = np.linspace(average_value - 5e-7, average_value + 5e-7, N)
        # Set the n(k) values for the simulations
        for i in range(len(sim_all)):
            n[i] = math.ceil((sim_all[i] - c[0]) / delta_c) + 1
            if n[i] > N:
                with open(
                    "/home/domagoj/domagoj_thesis/project/try/my_csvs/base_error.csv",
                    "a",
                    newline="",
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        np.block(
                            [
                                sim_all,
                                np.asarray(
                                    [
                                        1,
                                        i,
                                        delta_c,
                                        c[0],
                                        c[N - 1],
                                        math.ceil((sim_all[i] - c[0]) / delta_c),
                                        (sim_all[i] - c[0]) / delta_c,
                                    ]
                                ),
                            ]
                        )
                    )

    # In case the simulations are sufficiently apart:
    else:
        delta_c = (max_all - min_all) / (N - 1)
        c = np.linspace(min_all, max_all, N)
        for i in range(len(sim_all)):
            if i == max_idx:
                n[i] = N
            elif i == min_idx:
                n[i] = 1
            else:
                n[i] = math.ceil((sim_all[i] - c[0]) / delta_c) + 1  
            if n[i] > N:
                n[i] = N

    n = n.astype(int)
    return N, delta_c, c, n


def get_spline_bases_gradient(sim_all, sy_all, r):
    """ Get gradient of the rescaled spline bases """
    K = len(sim_all)

    N = math.ceil(r * K)

    min_idx = max_idx = 0
    # FIXME WE STILL have a problem here, what if there are multiple
    # simulations with same value y_max or y_min?... Which one?
    for idx in range(len(sim_all)):
        if sim_all[idx] > sim_all[max_idx]:
            max_idx = idx
        if sim_all[idx] < sim_all[min_idx]:
            min_idx = idx

    if sim_all[max_idx] - sim_all[min_idx] < 1e-6:
        delta_c_dot = 0
        c_dot = np.full(N, (sy_all[max_idx] - sy_all[min_idx]) / 2)
    else:
        delta_c_dot = (sy_all[max_idx] - sy_all[min_idx]) / (N - 1)
        c_dot = np.linspace(sy_all[min_idx], sy_all[max_idx], N)

    return delta_c_dot, c_dot


def get_sigma_for_group(gr: float, sigma: List[np.ndarray], simulation_indices: List):
    """ Get noise parameters for a group """
    gr = int(gr) - 1
    sim_length = 0
    for condition_indx in range(len(simulation_indices[gr])):
        sim_length += len(simulation_indices[gr][condition_indx])
    sigma_for_group = -np.ones((sim_length))

    current_indx = 0
    for condition_indx in range(len(simulation_indices[gr])):
        for time_indx in simulation_indices[gr][condition_indx]:
            sigma_for_group[current_indx] = sigma[condition_indx][time_indx][gr]
            current_indx += 1
    return sigma_for_group


def get_monotonicity_measure(quantitative_data, sim_all):
    """ Get monotonicity measure by calculating inversions """
    quantitative_data['simulation']=sim_all
    quantitative_data = quantitative_data.sort_values(by=['measurement'])
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
     #    print(quantitative_data)
    inversions=0
    ordered_simulation = quantitative_data.simulation.values
    measurement = quantitative_data.measurement.values
    for i in range(len(ordered_simulation)):
        for j in range(i+1, len(ordered_simulation)):
            if(ordered_simulation[i]>ordered_simulation[j]):
                inversions+=1
            elif(ordered_simulation[i]==ordered_simulation[j] and measurement[i]!=measurement[j]):
                inversions+=1
    #print(inversions)
    return inversions
