from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, lsq_linear, minimize
from scipy.special import betainc, betaln, xlogy

try:
    import fides
except ImportError:
    fides = None

from ...C import (
    BETA_A_BOUNDS,
    BETA_B_BOUNDS,
    BETA_BETA_STAR,
    BETA_KAPPA,
    CURRENT_SIMULATION,
    DATAPOINTS,
    EXPDATA_MASK,
    FAMILY_BETA_CDF,
    FAMILY_SPLINE,
    FUNCTION_FAMILIES,
    FUNCTION_FAMILY,
    INNER_NOISE_PARS,
    MAX_DATAPOINT,
    MIN_DATAPOINT,
    MIN_DIFF_FACTOR,
    MIN_SIM_RANGE,
    N_BETA_PARS,
    N_SPLINE_PARS,
    NUM_DATAPOINTS,
    OPTIMIZE_NOISE,
    REGULARIZATION_FACTOR,
    REGULARIZE_SPLINE,
    RSS_REL_FLOOR,
    SCIPY_FUN,
    SCIPY_SUCCESS,
    SCIPY_X,
    InnerParameterType,
)
from ..base_solver import InnerSolver
from .parameter import SplineInnerParameter
from .problem import SemiquantProblem

try:
    from amici.petab.parameter_mapping import ParameterMapping
except ImportError:
    pass

# Returned by the beta objective where the shapes give a degenerate design, so the optimizer walks away
# from that corner rather than seeing a non-finite value.
BETA_BIG_NLLH = 1e9


class SemiquantInnerSolver(InnerSolver):
    """Solver of the inner subproblem of spline approximation for nonlinear-monotone data.

    Options
    -------
    min_diff_factor:
        Determines the minimum difference between two consecutive spline
        as ``min_diff_factor * (measurement_range) / n_spline_pars``.
        Default is 1/2.
    """

    def __init__(self, options: dict = None):
        self.options = {
            **self.get_default_options(),
            **(options or {}),
        }
        self.validate_options()

    def validate_options(self):
        """Validate the current options dictionary."""
        if not isinstance(self.options[MIN_DIFF_FACTOR], float):
            raise TypeError(f"{MIN_DIFF_FACTOR} must be of type float.")
        elif self.options[MIN_DIFF_FACTOR] < 0:
            raise ValueError(f"{MIN_DIFF_FACTOR} must not be negative.")

        elif not isinstance(self.options[REGULARIZE_SPLINE], bool):
            raise TypeError(f"{REGULARIZE_SPLINE} must be of type bool.")
        if self.options[REGULARIZE_SPLINE]:
            if not isinstance(self.options[REGULARIZATION_FACTOR], float):
                raise TypeError(
                    f"{REGULARIZATION_FACTOR} must be of type float."
                )
            elif self.options[REGULARIZATION_FACTOR] < 0:
                raise ValueError(
                    f"{REGULARIZATION_FACTOR} must not be negative."
                )

        if self.options[FUNCTION_FAMILY] not in FUNCTION_FAMILIES:
            raise ValueError(
                f"{FUNCTION_FAMILY} must be one of {FUNCTION_FAMILIES}, "
                f"got {self.options[FUNCTION_FAMILY]!r}."
            )
        if self.options[FUNCTION_FAMILY] == FAMILY_BETA_CDF:
            if not isinstance(self.options[BETA_KAPPA], float):
                raise TypeError(f"{BETA_KAPPA} must be of type float.")
            elif self.options[BETA_KAPPA] < 1.0:
                raise ValueError(
                    f"{BETA_KAPPA} must be at least 1, so that Q bounds the simulations."
                )
            if not isinstance(self.options[BETA_BETA_STAR], float):
                raise TypeError(f"{BETA_BETA_STAR} must be of type float.")
            elif self.options[BETA_BETA_STAR] <= 0.0:
                raise ValueError(f"{BETA_BETA_STAR} must be positive.")

        for key in self.options:
            if key not in self.get_default_options():
                raise ValueError(f"Unknown SplineInnerSolver option {key}.")

    def solve(
        self,
        problem: SemiquantProblem,
        sim: list[np.ndarray],
        amici_sigma: list[np.ndarray],
    ) -> list:
        """Get results for every group (inner optimization problem).

        Parameters
        ----------
        problem:
            InnerProblem from pyPESTO hierarchical.
        sim:
            Simulations from AMICI.
        amici_sigma:
            List of sigmas from AMICI.

        Returns
        -------
        List of optimization results of the inner subproblem.
        """
        inner_results = []
        for group in problem.get_groups_for_xs(InnerParameterType.SEMIQUANT):
            group_dict = problem.groups[group]
            group_dict[CURRENT_SIMULATION] = extract_expdata_using_mask(
                expdata=sim, mask=group_dict[EXPDATA_MASK]
            )

            # If the noise parameters are optimized in the outer problem,
            # extract them from amici return data.
            if not group_dict[OPTIMIZE_NOISE]:
                group_dict[INNER_NOISE_PARS] = extract_expdata_using_mask(
                    expdata=amici_sigma, mask=group_dict[EXPDATA_MASK]
                )[0]

            # Dispatch on the recording family BEFORE _optimize_spline: its default path
            # (_solve_bounded_ls) is a linear least squares, which would "successfully" return the
            # optimum of a 2-knot spline for a beta group instead of failing.
            if self.options[FUNCTION_FAMILY] == FAMILY_BETA_CDF:
                inner_result_for_group = self._optimize_beta(
                    inner_parameters=problem.get_free_xs_for_group(group),
                    group_dict=group_dict,
                )
            else:
                inner_result_for_group = self._optimize_spline(
                    inner_parameters=problem.get_free_xs_for_group(group),
                    group_dict=group_dict,
                )

            inner_results.append(inner_result_for_group)
            save_inner_parameters_to_inner_problem(
                inner_problem=problem,
                s=inner_result_for_group[SCIPY_X],
                group=group,
            )
        return inner_results

    @staticmethod
    def calculate_obj_function(x_inner_opt: list):
        """Calculate the inner objective function value.

        Calculates the inner objective function value from a list of inner
        optimization results returned from `_optimize_spline`.

        Parameters
        ----------
        x_inner_opt:
            List of optimization results

        Returns
        -------
        Inner objective function value.
        """
        if False in (
            x_inner_opt[idx][SCIPY_SUCCESS] for idx in range(len(x_inner_opt))
        ):
            obj = np.inf
            warnings.warn(
                "Inner optimization failed.",
                stacklevel=2,
            )
        else:
            obj = np.sum(
                [
                    x_inner_opt[idx][SCIPY_FUN]
                    for idx in range(len(x_inner_opt))
                ]
            )
        return obj

    def check_if_can_use_gradients_same_plists(
        self,
        sy: list[np.ndarray],
        parameter_mapping: ParameterMapping,
        par_opt_ids: list,
        par_sim_ids: list,
        par_edatas_indices: list,
    ):
        """Check if gradients can be calculated with the assumption of same plists."""
        par_edatas_indices = [
            {par_edata_idx: idx for idx, par_edata_idx in enumerate(par_edata)}
            for par_edata in par_edatas_indices
        ]
        par_sim_ids = {
            par_sim_id: idx for idx, par_sim_id in enumerate(par_sim_ids)
        }
        par_opt_ids = {
            par_opt_id: idx for idx, par_opt_id in enumerate(par_opt_ids)
        }
        mappings_across_conditions = []

        for condition_map_sim_var in [
            cond_par_map.map_sim_var for cond_par_map in parameter_mapping
        ]:
            mapping_for_condition = {}
            for par_sim, par_opt in condition_map_sim_var.items():
                if not isinstance(par_opt, str):
                    continue
                elif par_opt not in par_opt_ids:
                    continue
                par_sim_idx = par_sim_ids[par_sim]
                par_opt_idx = par_opt_ids[par_opt]
                par_amici_rdata_idx = [
                    par_edata_indices[par_sim_idx]
                    for par_edata_indices in par_edatas_indices
                ]
                if all(
                    par_amici_rdata_id == par_amici_rdata_idx[0]
                    for par_amici_rdata_id in par_amici_rdata_idx
                ):
                    par_amici_rdata_idx = par_amici_rdata_idx[0]
                mapping_for_condition[par_opt_idx] = par_amici_rdata_idx
            mappings_across_conditions.append(mapping_for_condition)

        if not all(
            mapping_for_condition == mappings_across_conditions[0]
            for mapping_for_condition in mappings_across_conditions
        ):
            raise ValueError(
                "Parameter mappings are different across conditions."
            )

        if len(mappings_across_conditions[0].keys()) == len(sy[0]):
            raise ValueError(
                "The number of parameters in the mappings is different from the number of sensitivities."
            )

        if not all(
            par_edata_indices == par_edatas_indices[0]
            for par_edata_indices in par_edatas_indices
        ):
            raise ValueError(
                "plists are not the same for all conditions. "
                "Cannot use calculate_gradients_same_plists to calculate gradients."
            )

    def calculate_gradients_same_plists(
        self,
        problem: SemiquantProblem,
        x_inner_opt: list[dict],
        sim: list[np.ndarray],
        amici_sigma: list[np.ndarray],
        sy: list[np.ndarray],
        amici_ssigma: list[np.ndarray],
        parameter_mapping: ParameterMapping,
        par_opt_ids: list,
        par_sim_ids: list,
        par_edatas_indices: list,
        snllh: np.ndarray,
    ):
        """Calculate gradients of the inner objective function.

        Vectorized gradient calculation assuming all conditions share the same
        parameter list (plist). Huge speedup for the Frohlich model (16K conditions).

        Parameters
        ----------
        problem:
            Semiquant inner problem.
        x_inner_opt:
            List of optimization results of the inner subproblem.
        sim:
            Model simulations.
        amici_sigma:
            Model noise parameters.
        sy:
            Model sensitivities.
        amici_ssigma:
            Model sigma sensitivities.
        parameter_mapping:
            Mapping of optimization to simulation parameters.
        par_opt_ids:
            Ids of outer optimization parameters.
        par_sim_ids:
            Ids of outer simulation parameters, includes fixed parameters.
        par_edatas_indices:
            The indices of the parameters in the edatas.
        snllh:
            A zero-initialized vector of the same length as ``par_opt_ids`` to
            store the gradients in. Will be modified in-place.

        Returns
        -------
        The gradients with respect to the outer parameters.
        """
        # restructure sensitivities to have parameter index as second index
        sy = [np.moveaxis(sy_cond, 1, 0) for sy_cond in sy]

        n_conditions = len(sy)
        n_parameters = len(sy[0])

        par_edatas_indices = [
            {par_edata_idx: idx for idx, par_edata_idx in enumerate(par_edata)}
            for par_edata in par_edatas_indices
        ]
        par_sim_ids = {
            par_sim_id: idx for idx, par_sim_id in enumerate(par_sim_ids)
        }
        par_opt_ids = {
            par_opt_id: idx for idx, par_opt_id in enumerate(par_opt_ids)
        }

        # Paired sensitivity-column and outer-parameter positions. Columns mapping to
        # an inner parameter or to a fixed/numeric value are excluded: they have no
        # outer gradient contribution.
        sim_positions: list[int] = []
        opt_positions: list[int] = []
        for par_sim, par_opt in parameter_mapping[0].map_sim_var.items():
            if not isinstance(par_opt, str):
                continue
            elif par_opt not in par_opt_ids:
                continue
            par_sim_idx = par_sim_ids[par_sim]
            par_opt_idx = par_opt_ids[par_opt]
            # A model parameter absent from some condition's plist has no sensitivity
            # column for it.
            if any(
                par_sim_idx not in par_edata_indices
                for par_edata_indices in par_edatas_indices
            ):
                continue
            par_amici_rdata_idx = [
                par_edata_indices[par_sim_idx]
                for par_edata_indices in par_edatas_indices
            ]
            if all(
                par_amici_rdata_id == par_amici_rdata_idx[0]
                for par_amici_rdata_id in par_amici_rdata_idx
            ):
                par_amici_rdata_idx = [par_amici_rdata_idx[0]]
            for rdata_idx in par_amici_rdata_idx:
                sim_positions.append(int(rdata_idx))
                opt_positions.append(int(par_opt_idx))
        sim_positions = np.asarray(sim_positions, dtype=int)
        opt_positions = np.asarray(opt_positions, dtype=int)

        sim_grad = np.zeros(n_parameters)

        for group in problem.get_groups_for_xs(InnerParameterType.SEMIQUANT):
            group_dict = problem.groups[group]
            mask = group_dict[EXPDATA_MASK]
            group_dict[CURRENT_SIMULATION] = extract_expdata_using_mask(
                expdata=sim, mask=mask
            )
            s = np.asarray(x_inner_opt[group - 1][SCIPY_X])
            N = group_dict[N_SPLINE_PARS]
            K = group_dict[NUM_DATAPOINTS]
            measurements = group_dict[DATAPOINTS]
            sigma = problem.groups[group][INNER_NOISE_PARS]
            sim_all = extract_expdata_using_mask(expdata=sim, mask=mask)

            # Extract sensitivities for this group across all parameters
            # shape: (n_parameters, n_datapoints_in_group)
            sy_all_across_pars = np.concatenate(
                [sy[i][:, mask[i]] for i in range(n_conditions)], axis=1
            )

            if self.options[FUNCTION_FAMILY] == FAMILY_BETA_CDF:
                sim_grad += _beta_group_gradient(
                    solver=self,
                    sim_all=sim_all,
                    sy_all=sy_all_across_pars,
                    measurements=measurements,
                    inner_pars=s,
                )
                continue

            delta_c, c, n = SemiquantInnerSolver._rescale_spline_bases(
                sim_all=group_dict[CURRENT_SIMULATION], N=N, K=K
            )

            # Calculate gradient of spline knots c and delta_c
            min_idx = np.argmin(sim_all)
            max_idx = np.argmax(sim_all)
            average_value = (sim_all[max_idx] + sim_all[min_idx]) / 2
            if sim_all[max_idx] - sim_all[min_idx] < MIN_SIM_RANGE:
                delta_c_dot = np.full(n_parameters, 0)
                if average_value < (MIN_SIM_RANGE / 2):
                    c_dot_matrix = np.full((n_parameters, N), 0)
                else:
                    c_dot_matrix = np.tile(
                        (sy_all_across_pars[:, max_idx] - sy_all_across_pars[:, min_idx]) / 2,
                        (N, 1),
                    ).T
            else:
                c_dot_matrix = np.linspace(
                    sy_all_across_pars[:, min_idx],
                    sy_all_across_pars[:, max_idx],
                    N,
                    axis=1,
                )
                delta_c_dot = (
                    sy_all_across_pars[:, max_idx] - sy_all_across_pars[:, min_idx]
                ) / (N - 1)

            # Vectorized gradient wrt simulations y
            n_indices = n - 1
            valid_indices = (n_indices > 0) & (n_indices < N)
            valid_n_indices = n_indices[valid_indices]

            y_c_diff = sim_all[valid_indices] - c[valid_n_indices - 1]
            s_values = s[valid_n_indices]
            sum_s = np.cumsum(s[:-1])[valid_n_indices - 1]
            y_dot_valid = sy_all_across_pars[:, valid_indices]
            c_dot_valid = c_dot_matrix[:, valid_n_indices - 1]
            scaled_delta_c_dot = delta_c_dot / delta_c**2

            term1 = y_c_diff * s_values / delta_c + sum_s - measurements[valid_indices]
            term2 = s_values * (
                (y_dot_valid - c_dot_valid) / delta_c
                - np.outer(scaled_delta_c_dot, y_c_diff)
            )

            sim_grad += np.dot(term2, term1) / sigma**2

        # np.add.at accumulates when several sim params map to the same opt param.
        np.add.at(snllh, opt_positions, sim_grad[sim_positions])

        return snllh

    def calculate_gradients(
        self,
        problem: SemiquantProblem,
        x_inner_opt: list[dict],
        sim: list[np.ndarray],
        amici_sigma: list[np.ndarray],
        sy: list[np.ndarray],
        amici_ssigma: list[np.ndarray],
        parameter_mapping: ParameterMapping,
        par_opt_ids: list,
        par_sim_ids: list,
        par_edatas_indices: list,
        snllh: np.ndarray,
    ):
        """Calculate gradients of the inner objective function.

        Calculates gradients of the objective function with respect to outer
        (dynamical) parameters.

        Parameters
        ----------
        problem:
            Optimal scaling inner problem.
        x_inner_opt:
            List of optimization results of the inner subproblem.
        sim:
            Model simulations.
        sigma:
            Model noise parameters.
        sy:
            Model sensitivities.
        parameter_mapping:
            Mapping of optimization to simulation parameters.
        par_opt_ids:
            Ids of outer otimization parameters.
        par_sim_ids:
            Ids of outer simulation parameters, includes fixed parameters.
        snllh:
            A zero-initialized vector of the same length as ``par_opt_ids`` to store the
            gradients in. Will be modified in-place.

        Returns
        -------
        The gradients with respect to the outer parameters.
        """
        already_calculated = set()

        for condition_map_sim_var in [
            cond_par_map.map_sim_var for cond_par_map in parameter_mapping
        ]:
            for par_sim, par_opt in condition_map_sim_var.items():
                if (
                    not isinstance(par_opt, str)
                    or par_opt in already_calculated
                ):
                    continue
                elif par_opt not in par_opt_ids:
                    continue
                else:
                    already_calculated.add(par_opt)
                par_sim_idx = par_sim_ids.index(par_sim)
                par_opt_idx = par_opt_ids.index(par_opt)
                grad = 0.0

                sy_for_outer_parameter = [
                    (
                        sy_cond[:, par_edata_indices.index(par_sim_idx), :]
                        if par_sim_idx in par_edata_indices
                        else np.zeros(sy_cond[:, 0, :].shape)
                    )
                    for sy_cond, par_edata_indices in zip(
                        sy, par_edatas_indices, strict=True
                    )
                ]
                ssigma_for_outer_parameter = [
                    (
                        ssigma_cond[:, par_edata_indices.index(par_sim_idx), :]
                        if par_sim_idx in par_edata_indices
                        else np.zeros(ssigma_cond[:, 0, :].shape)
                    )
                    for ssigma_cond, par_edata_indices in zip(
                        amici_ssigma, par_edatas_indices, strict=True
                    )
                ]

                for group_idx, group in enumerate(
                    problem.get_groups_for_xs(InnerParameterType.SEMIQUANT)
                ):
                    # Get the reformulated spline parameters
                    s = np.asarray(x_inner_opt[group_idx][SCIPY_X])
                    group_dict = problem.groups[group]

                    measurements = group_dict[DATAPOINTS]
                    sigma = group_dict[INNER_NOISE_PARS]
                    sim_all = group_dict[CURRENT_SIMULATION]
                    N = group_dict[N_SPLINE_PARS]
                    K = group_dict[NUM_DATAPOINTS]

                    sy_all = extract_expdata_using_mask(
                        expdata=sy_for_outer_parameter,
                        mask=group_dict[EXPDATA_MASK],
                    )
                    ssigma_all = extract_expdata_using_mask(
                        expdata=ssigma_for_outer_parameter,
                        mask=group_dict[EXPDATA_MASK],
                    )

                    if self.options[FUNCTION_FAMILY] == FAMILY_BETA_CDF:
                        # This path walks one outer parameter at a time, so sy_all is 1-D here while
                        # the shared helper is written for (n_parameters, n_datapoints).
                        grad += _beta_group_gradient(
                            solver=self,
                            sim_all=sim_all,
                            sy_all=sy_all[np.newaxis, :],
                            measurements=measurements,
                            inner_pars=s,
                        )[0]
                        continue

                    delta_c, c, n = self._rescale_spline_bases(
                        sim_all=sim_all, N=N, K=K
                    )
                    delta_c_dot, c_dot = calculate_spline_bases_gradient(
                        sim_all=sim_all, sy_all=sy_all, N=N
                    )

                    # For the reformulated problem, mu can be calculated
                    # as the inner gradient at the optimal point s.
                    mu = _calculate_nllh_gradient_for_group(
                        s=s,
                        sim_all=sim_all,
                        measurements=measurements,
                        N=N,
                        delta_c=delta_c,
                        c=c,
                        n=n,
                        regularization_factor=self.options[
                            REGULARIZATION_FACTOR
                        ],
                        regularize_spline=self.options[REGULARIZE_SPLINE],
                        group_dict=group_dict,
                    )
                    min_meas = group_dict[MIN_DATAPOINT]
                    max_meas = group_dict[MAX_DATAPOINT]
                    min_diff = self._get_minimal_difference(
                        measurement_range=max_meas - min_meas,
                        N=N,
                        min_diff_factor=self.options[MIN_DIFF_FACTOR],
                    )

                    # If the spline parameter is at its boundary, the
                    # corresponding Lagrangian multiplier mu is set to 0.
                    min_diff_all = np.full(N, min_diff)
                    min_diff_all[0] = 0.0
                    mu = np.asarray(
                        [
                            (
                                mu[i]
                                if np.isclose(s[i] - min_diff_all[i], 0)
                                else 0
                            )
                            for i in range(len(s))
                        ]
                    )

                    # Calculate the (dJ_dy * dy_dtheta) term:
                    dy_grad_term = calculate_dy_term(
                        sim_all=sim_all,
                        sy_all=sy_all,
                        measurements=measurements,
                        s=s,
                        N=N,
                        delta_c=delta_c,
                        delta_c_dot=delta_c_dot,
                        c=c,
                        c_dot=c_dot,
                        n=n,
                    )

                    # Calculate the (dJ_dsigma^2 * dsigma^2_dtheta) term:
                    if not group_dict[OPTIMIZE_NOISE]:
                        residual_squared = _calculate_residuals_for_group(
                            s=s,
                            sim_all=sim_all,
                            measurements=measurements,
                            N=N,
                            delta_c=delta_c,
                            c=c,
                            n=n,
                        )
                        dJ_dsigma2 = (
                            K / (2 * sigma**2) - residual_squared / sigma**4
                        )
                        dsigma2_dtheta = ssigma_all[0] * sigma
                        dsigma_grad_term = dJ_dsigma2 * dsigma2_dtheta
                    # If we optimize the noise hierarchically,
                    # the last term (dJ_dsigma^2 * dsigma^2_dtheta) is always 0
                    # since the sigma is optimized such that dJ_dsigma2=0.
                    else:
                        dsigma_grad_term = 0.0

                    # Combine all terms to get the complete gradient contribution
                    grad += dy_grad_term / sigma**2 + dsigma_grad_term

                snllh[par_opt_idx] = grad

        return snllh

    @staticmethod
    def get_default_options() -> dict:
        """Return default options for solving the inner problem."""
        options = {
            MIN_DIFF_FACTOR: 0.0,
            REGULARIZE_SPLINE: False,
            REGULARIZATION_FACTOR: 0.0,
            FUNCTION_FAMILY: FAMILY_SPLINE,
            # kappa = 1.05 leaves 5% headroom above the (already >= max) LSE anchor. It barely affects
            # the theta inference -- fit quality moves < 0.28 nllh across kappa in [1, 2] -- but it DOES
            # set how much of [0,1] the data occupy, so it is part of what the fitted (a, b) MEAN. Any
            # cross-study comparison of shapes has to quote the same kappa.
            BETA_KAPPA: 1.05,
            BETA_BETA_STAR: 31.0,
        }
        return options

    def _optimize_spline(
        self,
        inner_parameters: list[SplineInnerParameter],
        group_dict: dict,
    ):
        """Run optimization for the inner problem.

        Parameters
        ----------
        inner_parameters:
            The spline inner parameters.
        group_dict:
            The group dictionary.
        """
        (
            distance_between_bases,
            spline_bases,
            intervals_per_sim,
        ) = self._rescale_spline_bases(
            sim_all=group_dict[CURRENT_SIMULATION],
            N=group_dict[N_SPLINE_PARS],
            K=group_dict[NUM_DATAPOINTS],
        )

        min_diff = self._get_minimal_difference(
            measurement_range=group_dict[MAX_DATAPOINT]
            - group_dict[MIN_DATAPOINT],
            N=group_dict[N_SPLINE_PARS],
            min_diff_factor=self.options[MIN_DIFF_FACTOR],
        )

        inner_options = self._get_inner_optimization_options(
            inner_parameters=inner_parameters,
            N=group_dict[N_SPLINE_PARS],
            min_meas=group_dict[MIN_DATAPOINT],
            max_meas=group_dict[MAX_DATAPOINT],
            min_diff=min_diff,
        )

        # Wrap the analytical optimization of sigma and
        # the regularization into the objective function
        def objective_function_wrapper(x):
            return _calculate_nllh_for_group(
                s=x,
                sim_all=group_dict[CURRENT_SIMULATION],
                measurements=group_dict[DATAPOINTS],
                N=group_dict[N_SPLINE_PARS],
                delta_c=distance_between_bases,
                c=spline_bases,
                n=intervals_per_sim,
                regularization_factor=self.options[REGULARIZATION_FACTOR],
                regularize_spline=self.options[REGULARIZE_SPLINE],
                group_dict=group_dict,
            )

        # Wrap the analytical optimization of sigma and
        # the regularization into the gradient function
        def inner_gradient_wrapper(x):
            return _calculate_nllh_gradient_for_group(
                s=x,
                sim_all=group_dict[CURRENT_SIMULATION],
                measurements=group_dict[DATAPOINTS],
                N=group_dict[N_SPLINE_PARS],
                delta_c=distance_between_bases,
                c=spline_bases,
                n=intervals_per_sim,
                regularization_factor=self.options[REGULARIZATION_FACTOR],
                regularize_spline=self.options[REGULARIZE_SPLINE],
                group_dict=group_dict,
            )

        # Without regularization the inner problem is a bound-constrained linear least squares, so it
        # can be solved exactly. With regularization the nllh is (K/2)log(RSS) + lambda*R(s), whose
        # optimum is not a least-squares optimum, and we fall through to L-BFGS-B.
        if not self.options[REGULARIZE_SPLINE]:
            exact = self._solve_bounded_ls(
                sim_all=group_dict[CURRENT_SIMULATION],
                measurements=group_dict[DATAPOINTS],
                N=group_dict[N_SPLINE_PARS],
                delta_c=distance_between_bases,
                c=spline_bases,
                n=intervals_per_sim,
                min_diff=min_diff,
                nllh=objective_function_wrapper,
                grad=inner_gradient_wrapper,
            )
            if exact is not None:
                return exact

        results = minimize(
            objective_function_wrapper,
            jac=inner_gradient_wrapper,
            **inner_options,
        )

        # The warm start is the previous theta's optimum, but the knot grid is re-anchored to the
        # current simulation range, so it can be badly mis-scaled. Retry once from the cold start.
        if not results.success:
            cold_options = self._get_inner_optimization_options(
                inner_parameters=inner_parameters,
                N=group_dict[N_SPLINE_PARS],
                min_meas=group_dict[MIN_DATAPOINT],
                max_meas=group_dict[MAX_DATAPOINT],
                min_diff=min_diff,
                force_cold_start=True,
            )
            if not np.allclose(cold_options["x0"], inner_options["x0"]):
                retry = minimize(
                    objective_function_wrapper,
                    jac=inner_gradient_wrapper,
                    **cold_options,
                )
                if retry.success or retry.fun < results.fun:
                    return retry

        return results

    def _optimize_beta(
        self,
        inner_parameters: list[SplineInnerParameter],
        group_dict: dict,
    ):
        """Run optimization for the inner problem, beta CDF family.

        Only the two shapes are optimized; offset, scale and sigma are concentrated out, so the
        objective is ``(K/2) log(RSS/K)`` with ``RSS = SST (1 - rho^2)``. Optimization is in
        ``(log a, log b)``, where the shape box is a rectangle.

        A degenerate group returns ``success=False`` rather than raising, so that
        ``calculate_obj_function`` turns it into an infinite objective and the outer optimizer moves
        away from that theta instead of the run crashing.
        """
        sim_all = group_dict[CURRENT_SIMULATION]
        measurements = group_dict[DATAPOINTS]
        K = len(measurements)
        failed = OptimizeResult(
            x=np.array([1.0, 1.0]), fun=np.inf, success=False
        )

        if K < N_BETA_PARS + 4 or len(np.unique(sim_all)) < 3:
            # Four mean parameters (offset, scale, a, b) plus residual degrees of freedom for sigma;
            # and with offset and scale free, two distinct simulations are fitted exactly by any
            # monotone curve, so the shapes carry no information.
            return failed
        y_centered = measurements - measurements.mean()
        sst = float(y_centered @ y_centered)
        if np.std(measurements) <= 1e-13 * max(
            float(np.max(np.abs(measurements))), 1.0
        ):
            # Relative, not `sst > 0`: K copies of one value leave sst ~ K eps^2 y^2, which passes an
            # absolute test, and the fit would then maximize correlation against rounding noise.
            return failed
        try:
            domain = self._rescale_beta_domain(
                sim_all, self.options[BETA_KAPPA], self.options[BETA_BETA_STAR]
            )
        except ValueError:
            return failed

        x_all = (sim_all - domain["L"]) / (domain["Q"] - domain["L"])
        x_unique, x_inverse = np.unique(x_all, return_inverse=True)
        lb = np.log([BETA_A_BOUNDS[0], BETA_B_BOUNDS[0]])
        ub = np.log([BETA_A_BOUNDS[1], BETA_B_BOUNDS[1]])

        def objective_function_wrapper(p):
            return _calculate_nllh_for_group_beta(
                p, x_unique, x_inverse, y_centered, sst, K
            )

        def inner_gradient_wrapper(p):
            return _calculate_nllh_shape_gradient_for_group_beta(
                p, x_unique, x_inverse, y_centered, sst, K
            )

        # Started at the affine point (a, b) = (1, 1), never warm-started: the domain re-anchors to the
        # current simulations, so a warm start would make the inner optimum path-dependent.
        results = _minimize_beta_shapes(
            objective_function_wrapper, inner_gradient_wrapper, np.zeros(2), lb, ub
        )
        a, b = np.exp(np.clip(results.x, lb, ub))
        return OptimizeResult(
            x=np.array([a, b]), fun=float(results.fun), success=True
        )

    @staticmethod
    def _solve_bounded_ls(
        sim_all: np.ndarray,
        measurements: np.ndarray,
        N: int,
        delta_c: float,
        c: np.ndarray,
        n: np.ndarray,
        min_diff: float,
        nllh,
        grad,
    ):
        """Solve the inner problem exactly as bound-constrained linear least squares.

        The fitted value is linear in s, so the design has D[k, :i] = 1 and
        D[k, i] = (y_k - c[i-1]) / delta_c for interval index i = n_k - 1.

        Returns an OptimizeResult whose ``fun`` is the nllh (not the residual sum, which callers would
        sum as the inner objective), or None on any degeneracy so the caller can fall back. ``jac`` is
        included so this path returns the same keys as the L-BFGS-B fallback.
        """
        try:
            if not np.isfinite(delta_c) or delta_c <= 0:
                return None
            D = np.zeros((len(sim_all), N))
            for k, (y_k, n_k) in enumerate(zip(sim_all, n, strict=True)):
                i = int(n_k) - 1
                if i <= 0:
                    D[k, 0] = 1.0
                elif i >= N:
                    D[k, :] = 1.0
                else:
                    D[k, :i] = 1.0
                    D[k, i] = (y_k - c[i - 1]) / delta_c
            lb = np.full(N, float(min_diff))
            lb[0] = 0.0
            res = lsq_linear(
                D,
                np.asarray(measurements, float),
                bounds=(lb, np.inf),
                method="bvls",
            )
            if not res.success or not np.all(np.isfinite(res.x)):
                return None
            fval = float(nllh(res.x))
            if not np.isfinite(fval):
                return None
            return OptimizeResult(
                x=np.asarray(res.x, float),
                fun=fval,
                jac=np.asarray(grad(res.x), float),
                success=True,
                status=0,
                message="exact bounded least squares",
            )
        except Exception:
            return None

    @staticmethod
    def _rescale_spline_bases(sim_all: np.ndarray, N: int, K: int):
        """Rescale the spline bases.

        Before the optimization of the spline parameters, we have to fix the
        spline bases to some values. We choose to scale them to the current
        simulation. In case of simulations that are very close to each other,
        we choose to scale closely around the average value of the simulations,
        to avoid numerical problems (as we often divide by delta_c).

        Parameters
        ----------
        sim_all:
            The current simulation.
        N:
            The number of spline parameters.
        K:
            The number of simulations.

        Returns
        -------
        distance_between_bases:
            The distance between the spline bases.
        spline_bases:
            The rescaled spline bases.
        intervals_per_sim:
            List of indices of intervals each simulation belongs to.
        """
        min_idx = np.argmin(sim_all)
        max_idx = np.argmax(sim_all)

        min_all = sim_all[min_idx]
        max_all = sim_all[max_idx]

        n = np.ones(K)

        # In case the simulation are very close to each other
        # or even collapse into a single point.
        if max_all - min_all < MIN_SIM_RANGE:
            average_value = (max_all + min_all) / 2
            delta_c = MIN_SIM_RANGE / (N - 1)
            if average_value < (MIN_SIM_RANGE / 2):
                c = np.linspace(0, MIN_SIM_RANGE, N)
            else:
                c = np.linspace(
                    average_value - (MIN_SIM_RANGE / 2),
                    average_value + (MIN_SIM_RANGE / 2),
                    N,
                )
            # Set the n(k) values for the simulations
            for i in range(len(sim_all)):
                n[i] = np.ceil((sim_all[i] - c[0]) / delta_c) + 1
                if n[i] > N:
                    n[i] = N
                    warnings.warn(
                        "Interval for a simulation has been set to a larger "
                        "value than the number of spline parameters.",
                        stacklevel=2,
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
                    n[i] = np.ceil((sim_all[i] - c[0]) / delta_c) + 1
                if n[i] > N:
                    n[i] = N

        n = n.astype(int)
        return delta_c, c, n

    @staticmethod
    def _rescale_beta_domain(
        sim_all: np.ndarray,
        kappa: float,
        beta_star: float,
    ) -> dict:
        """Rescale the beta CDF domain to the simulations, via smooth bounds on both ends.

            Q = kappa * LSE_beta(q)        LSE_beta(q)     =  (1/beta) log sum_i exp( beta q_i)
            L =         SOFTMIN_beta(q)    SOFTMIN_beta(q) = -(1/beta) log sum_i exp(-beta q_i)
            x = (q - L) / (Q - L)          beta = beta_star / rms(q)

        LSE >= max(q) and SOFTMIN <= min(q) for any beta > 0, so x is in [0, 1] by construction whenever
        kappa >= 1. beta is normalized by rms(q), not max(q): max(q) would put argmax into dbeta/dtheta
        and so into both anchors' gradients. See METHOD_NOTES 2.3b.

        Returns
        -------
        Dict with the anchors ``Q``, ``L``; the softmax and softmin weights ``w`` = dLSE/dq,
        ``v`` = dSOFTMIN/dq; the beta chain factors ``dlse_dbeta`` <= 0, ``dsmin_dbeta`` >= 0; and
        ``beta``, ``rms``, ``drms_dsim`` = drms/dq. Everything both gradient paths need, computed once.
        """
        sim_all = np.asarray(sim_all, dtype=float)
        n = sim_all.size
        rms = float(np.sqrt(np.mean(sim_all**2)))
        if not rms > 0.0:
            raise ValueError(
                "All simulations of a semiquantitative group are zero, so the beta CDF "
                "domain cannot be anchored."
            )
        beta = beta_star / rms

        # Shifted by the extremum before exponentiating, so no term can overflow for large beta.
        mx, mn = float(sim_all.max()), float(sim_all.min())
        e_hi = np.exp(beta * (sim_all - mx))
        sum_hi = float(e_hi.sum())
        lse = mx + np.log(sum_hi) / beta
        e_lo = np.exp(-beta * (sim_all - mn))
        sum_lo = float(e_lo.sum())
        smin = mn - np.log(sum_lo) / beta

        Q, L = kappa * lse, smin
        if not Q > L:
            raise ValueError(f"Collapsed beta CDF domain: Q = {Q!r} <= L = {L!r}.")

        w = e_hi / sum_hi
        v = e_lo / sum_lo
        # As entropies rather than (sum_i w_i q_i - LSE)/beta: the two forms are equal by the Gibbs
        # identity H(w) = beta (LSE - sum_i w_i q_i), but that difference is only ~3% of LSE and loses
        # ~1.5 digits to cancellation. xlogy gives 0 for the underflowed weights.
        return {
            "Q": Q,
            "L": L,
            "w": w,
            "v": v,
            "dlse_dbeta": float(np.sum(xlogy(w, w))) / beta**2,
            "dsmin_dbeta": -float(np.sum(xlogy(v, v))) / beta**2,
            "beta": beta,
            "rms": rms,
            "drms_dsim": sim_all / (n * rms),
        }

    def _get_minimal_difference(
        self,
        measurement_range: float,
        N: int,
        min_diff_factor: float,
    ):
        """Return minimal parameter difference for spline parameters."""
        return min_diff_factor * measurement_range / N

    def _get_inner_optimization_options(
        self,
        inner_parameters: list[SplineInnerParameter],
        N: int,
        min_meas: float,
        max_meas: float,
        min_diff: float,
        force_cold_start: bool = False,
    ) -> dict:
        """Return default options for scipy optimizer.

        Returns inner subproblem optimization options including startpoint
        and optimization bounds or constraints, dependent on solver method.

        Parameters
        ----------
        inner_parameters:
            Inner parameters of the spline group.
        N:
            Number of spline parameters.
        min_meas:
            Minimal measurement value.
        max_meas:
            Maximal measurement value.
        min_diff:
            Minimal difference between spline parameters.
        """
        range_all = max_meas - min_meas

        constraint_min_diff = np.full(N, min_diff)
        constraint_min_diff[0] = 0

        last_opt_values = np.asarray([x.value for x in inner_parameters])

        if (last_opt_values > 0).any() and not force_cold_start:
            x0 = last_opt_values
        # In case this is the first inner optimization, initialize the
        # spline parameters to a linear function with a symmetric 60%
        # larger range than the measurement range.
        else:
            x0 = np.full(
                N,
                (
                    max_meas
                    + 0.3 * range_all
                    - np.max([min_meas - 0.3 * range_all, 0])
                )
                / (N - 1),
            )
            x0[0] = np.max([min_meas - 0.3 * range_all, 0])

        from scipy.optimize import Bounds

        inner_options = {
            "x0": x0,
            "method": "L-BFGS-B",
            "options": {"disp": None},
            "bounds": Bounds(lb=constraint_min_diff),
        }

        return inner_options


def _floor_rss(rss: float, sst: float) -> float:
    """Floor RSS relative to SST.

    A near-exact fit would otherwise send ``(K/2) log(RSS)`` to -inf and the gradient prefactor
    ``K/(2 RSS)`` to infinity. The floor is relative so that the objective and gradient stay invariant
    under a rescaling of the measurements. The SAME floor must be used by objective and gradient, or
    they describe different functions and every finite-difference check silently fails.
    """
    return max(rss, RSS_REL_FLOOR * sst)


def _floor_residuals_squared(
    residuals_squared: float, measurements: np.ndarray
) -> float:
    """Apply the RSS floor in the spline convention, where residuals are RSS/2.

    Floors relative to the total sum of squares of the measurements, so the bound is invariant under a
    rescaling of the data. Objective and gradient must both apply it, or they describe different
    functions.
    """
    centered = measurements - measurements.mean()
    return 0.5 * _floor_rss(2 * residuals_squared, float(centered @ centered))


def _calculate_nllh_for_group_beta(
    p: np.ndarray,
    x_unique: np.ndarray,
    x_inverse: np.ndarray,
    y_centered: np.ndarray,
    sst: float,
    K: int,
):
    """Beta CDF inner objective at shapes ``p = (log a, log b)``, offset/scale/sigma concentrated out.

    ``RSS = SST (1 - rho^2(z, y))`` with ``z_i = I_{x_i}(a, b)``, so the shape problem is correlation
    maximization. betainc is evaluated on the unique simulations and expanded, which is exact.

    Carries the Gaussian normalization, as ``_calculate_nllh_for_group`` does, so that values are
    comparable across observation models. It is constant in ``p`` and so does not enter the gradient.
    """
    a, b = np.exp(p)
    with np.errstate(all="ignore"):
        z = betainc(a, b, x_unique)[x_inverse]
    if not np.all(np.isfinite(z)):
        return BETA_BIG_NLLH
    z_centered = z - z.mean()
    d = float(z_centered @ z_centered)
    if d <= 1e-28:
        return BETA_BIG_NLLH
    rss = _floor_rss(sst - float(z_centered @ y_centered) ** 2 / d, sst)
    return 0.5 * K * (np.log(rss / K) + np.log(2 * np.pi) + 1)


def _calculate_nllh_shape_gradient_for_group_beta(
    p: np.ndarray,
    x_unique: np.ndarray,
    x_inverse: np.ndarray,
    y_centered: np.ndarray,
    sst: float,
    K: int,
):
    """Gradient of the beta inner objective with respect to ``(log a, log b)``.

    With ``n = z_c . y_c`` and ``d = z_c . z_c``,
    ``dRSS/dp = -2 (n/d) (dz . y_c) + 2 (n/d)^2 (z_c . dz)`` and ``dnllh/dp = (K/2) dRSS/dp / RSS``.
    ``dI/da`` and ``dI/db`` come from central differences on betainc itself, which reaches ~1e-10 --
    differencing the composed objective instead would only reach ~1e-8.
    """
    a, b = np.exp(p)
    rel = 1e-5
    ha, hb = a * rel, b * rel
    with np.errstate(all="ignore"):
        z = betainc(a, b, x_unique)
        dz_da = (betainc(a + ha, b, x_unique) - betainc(a - ha, b, x_unique)) / (2 * ha)
        dz_db = (betainc(a, b + hb, x_unique) - betainc(a, b - hb, x_unique)) / (2 * hb)
    if not (
        np.all(np.isfinite(z))
        and np.all(np.isfinite(dz_da))
        and np.all(np.isfinite(dz_db))
    ):
        return np.zeros(2)
    z = z[x_inverse]
    # Chain to the log-parameters: dz/d(log a) = a dI/da.
    derivatives = (a * dz_da[x_inverse], b * dz_db[x_inverse])
    z_centered = z - z.mean()
    d = float(z_centered @ z_centered)
    if d <= 1e-28:
        return np.zeros(2)
    n = float(z_centered @ y_centered)
    rss_raw = sst - n * n / d
    rss = _floor_rss(rss_raw, sst)
    if rss_raw < rss:
        # On the floored branch the objective is constant, so its derivative is zero.
        return np.zeros(2)
    gradient = np.array(
        [
            -2 * n / d * float(dz @ y_centered)
            + 2 * n * n / (d * d) * float(z_centered @ dz)
            for dz in derivatives
        ]
    )
    gradient *= 0.5 * K / rss
    return gradient if np.all(np.isfinite(gradient)) else np.zeros(2)


def _minimize_beta_shapes(nllh, grad, x0: np.ndarray, lb: np.ndarray, ub: np.ndarray):
    """Minimize the beta inner objective over the shape box.

    fides is trust-region reflective: the bounds enter the step computation, so it cannot stop at a box
    corner where the projected gradient is trivially zero. That is exactly how L-BFGS-B fails on this
    problem, at every value of its finite-difference step (METHOD_NOTES 2.9e). Without fides installed,
    fall back to bounded Nelder-Mead plus a Newton polish, which is equally reliable at finding the
    basin but needs the polish to reach a stationary point.
    """
    if fides is not None:
        optimizer = fides.Optimizer(
            lambda p: (nllh(p), grad(p)),
            ub=ub,
            lb=lb,
            verbose=logging.ERROR,
            hessian_update=fides.BFGS(),
            options={
                fides.Options.MAXITER: 200,
                fides.Options.FATOL: 1e-14,
                fides.Options.GATOL: 1e-10,
            },
        )
        _, x, _, _ = optimizer.minimize(x0)
        # Polished even though fides converged: it stops on its own criteria, which the outer gradient
        # does not care about. The ENVELOPE THEOREM needs a stationary point, and the outer gradient
        # error is FIRST order in the inner gradient while being invisible in the objective value.
        # Measured here: fides stopped at |g| = 2.4e-3 within 4e-8 of the optimal value, and that alone
        # put 1.9e-3 into the outer gradient; polishing to |g| = 3.4e-9 left 9.1e-7. ~10 extra
        # evaluations for a 2000x more accurate gradient.
        return _newton_polish_beta_shapes(np.asarray(x, float), nllh, grad, lb, ub)

    # scipy's default Nelder-Mead simplex uses a 5% relative step but falls back to 0.00025 absolute
    # for a coordinate that is exactly zero -- and log(1) is exactly zero, so the simplex at the (1,1)
    # start would be born a thousand times too small.
    simplex = np.clip(
        np.vstack([x0, x0 + [0.4, 0.0], x0 + [0.0, 0.4]]), lb, ub
    )
    result = minimize(
        nllh,
        x0,
        method="Nelder-Mead",
        bounds=list(zip(lb, ub, strict=True)),
        options={"initial_simplex": simplex, "xatol": 1e-8, "fatol": 1e-11},
    )
    return _newton_polish_beta_shapes(result.x, nllh, grad, lb, ub)


def _newton_polish_beta_shapes(p, nllh, grad, lb, ub, iterations: int = 2):
    """Give Nelder-Mead the stationarity it lacks, on the free set only, accepting only on descent.

    Nelder-Mead stops on simplex size, which is not a stationarity certificate, and the envelope
    theorem needs one. The problem is 2-D, so an exact Newton step costs four gradient evaluations.
    """
    p = np.clip(np.asarray(p, float), lb, ub)
    fval = nllh(p)
    for _ in range(iterations):
        g = grad(p)
        free = [
            i
            for i in range(len(p))
            if not (
                (p[i] <= lb[i] + 1e-9 and g[i] > 0)
                or (p[i] >= ub[i] - 1e-9 and g[i] < 0)
            )
        ]
        if not free:
            break
        hessian = np.empty((len(p), len(p)))
        step_h = 1e-5
        for j in range(len(p)):
            offset = np.zeros(len(p))
            offset[j] = step_h
            hessian[:, j] = (grad(p + offset) - grad(p - offset)) / (2 * step_h)
        hessian = 0.5 * (hessian + hessian.T)
        try:
            step = np.linalg.solve(hessian[np.ix_(free, free)], -g[free])
        except np.linalg.LinAlgError:
            break
        candidate = p.copy()
        candidate[free] = p[free] + step
        candidate = np.clip(candidate, lb, ub)
        fval_candidate = nllh(candidate)
        if fval_candidate > fval:
            break
        p, fval = candidate, fval_candidate
    return OptimizeResult(x=p, fun=float(fval), success=True)


def _calculate_nllh_for_group(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
    regularization_factor: float,
    regularize_spline: bool,
    group_dict: dict,
) -> float:
    """Calculate the negative log-likelihood for the group.

    Combines the sum of squared residuals, the noise parameter,
    and the regularization term to the negative log-likelihood.

    Parameters
    ----------
    s:
        Reformulated inner spline parameters.
    sim_all:
        Simulations for the group.
    measurements:
        Measurements for the group.
    N:
        Number of spline bases.
    delta_c:
        Distance between two spline bases.
    c:
        Spline bases.
    n:
        Indices of the spline bases.
    regularization_factor:
        Regularization factor.
    regularize_spline:
        Whether to regularize the spline.
    group_dict:
        Dictionary containing the group information.

    Returns
    -------
    Negative log-likelihood.
    """
    # Calculate residuals
    residuals_squared = _calculate_residuals_for_group(
        s=s,
        sim_all=sim_all,
        measurements=measurements,
        N=N,
        delta_c=delta_c,
        c=c,
        n=n,
    )
    K = len(sim_all)

    # Calculate sigma
    if group_dict[OPTIMIZE_NOISE]:
        # sigma is concentrated out here, so a near-exact fit would send log(sigma^2) to -inf and the
        # prefactor 1/sigma^2 to infinity; clamp it from below. The residual term below deliberately
        # keeps the UNCLAMPED residuals: once sigma is clamped it no longer cancels to K/2, and that
        # is what keeps the objective responsive to the fit -- which in turn is why the existing
        # gradient formulas, here and in the outer path, stay correct with no floored-branch special
        # case. The clamped sigma depends only on the measurements, so it is constant in theta and s.
        sigma = _calculate_sigma_for_group(
            residuals_squared=_floor_residuals_squared(
                residuals_squared, measurements
            ),
            n_datapoints=K,
        )
        group_dict[INNER_NOISE_PARS] = sigma
    else:
        sigma = group_dict[INNER_NOISE_PARS]

    # Calculate regularization term
    if regularize_spline:
        regularization_term = _calculate_regularization_for_group(
            s=s,
            N=N,
            c=c,
            regularization_factor=regularization_factor,
        )
    else:
        regularization_term = 0.0

    # Combine all terms into the negative log-likelihood
    nllh = (
        0.5 * np.log(2 * np.pi * sigma**2) * K
        + residuals_squared / (sigma**2)
        + regularization_term
    )
    return nllh


def _calculate_nllh_gradient_for_group(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
    regularization_factor: float,
    regularize_spline: bool,
    group_dict: dict,
) -> np.ndarray:
    """Calculate the gradient of the nllh wrt. spline differences s for the group.

    Combines the gradient of the sum of squared residuals and the gradient of the
    regularization term to the gradient of the negative log-likelihood.

    Parameters
    ----------
    s:
        Reformulated inner spline parameters.
    sim_all:
        Simulations for the group.
    measurements:
        Measurements for the group.
    N:
        Number of spline bases.
    delta_c:
        Distance between two spline bases.
    c:
        Spline bases.
    n:
        Indices of the spline bases.
    regularization_factor:
        Regularization factor.
    regularize_spline:
        Whether to regularize the spline.
    group_dict:
        Dictionary containing the group information.

    Returns
    -------
    Gradient of the negative log-likelihood wrt. spline differences s.
    """
    # Calculate gradient of residuals
    residuals_squared_gradient = _calculate_residuals_gradient_for_group(
        s=s,
        sim_all=sim_all,
        measurements=measurements,
        N=N,
        delta_c=delta_c,
        c=c,
        n=n,
    )

    # Calculate sigma
    if group_dict[OPTIMIZE_NOISE]:
        residuals_squared = _calculate_residuals_for_group(
            s=s,
            sim_all=sim_all,
            measurements=measurements,
            N=N,
            delta_c=delta_c,
            c=c,
            n=n,
        )
        # The same clamp the objective applies; see _calculate_nllh_for_group.
        sigma = _calculate_sigma_for_group(
            residuals_squared=_floor_residuals_squared(
                residuals_squared, measurements
            ),
            n_datapoints=len(sim_all),
        )
        group_dict[INNER_NOISE_PARS] = sigma
    else:
        sigma = group_dict[INNER_NOISE_PARS]

    # Calculate gradient of regularization term
    if regularize_spline:
        regularization_term_gradient = (
            _calculate_regularization_gradient_for_group(
                s=s,
                N=N,
                c=c,
                regularization_factor=regularization_factor,
            )
        )
    else:
        regularization_term_gradient = np.zeros_like(s)

    # Combine all terms into the gradient of the negative log-likelihood
    nllh_gradient = (
        residuals_squared_gradient / (sigma**2) + regularization_term_gradient
    )
    return nllh_gradient


def _calculate_sigma_for_group(
    residuals_squared: float,
    n_datapoints: int,
):
    """Calculate the noise parameter sigma.

    Parameters
    ----------
    residuals_squared:
        The sum of squared residuals divided by 2.
    n_datapoints:
        The number of datapoints.
    """
    sigma = np.sqrt(2 * residuals_squared / n_datapoints)

    return sigma


def _calculate_residuals_for_group(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Residuals squared for reformulated inner spline problem.

    Equal to 1/2 * sum_k (tilde{z}_k - z_k)^2
    """
    obj = 0

    for y_k, z_k, n_k in zip(sim_all, measurements, n, strict=True):
        i = n_k - 1
        sum_s = 0
        sum_s = np.sum(s[:i])
        if i == 0:
            obj += (z_k - s[i]) ** 2
        elif i == N:
            obj += (z_k - sum_s) ** 2
        else:
            obj += (z_k - (y_k - c[i - 1]) * s[i] / delta_c - sum_s) ** 2
    obj = obj / 2
    return obj


def _calculate_residuals_gradient_for_group(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Gradient of the residuals with respect to the spline differences s_i for a group."""

    gradient = np.zeros(N)

    for y_k, z_k, n_k in zip(sim_all, measurements, n, strict=True):
        sum_s = 0
        i = n_k - 1  # just the iterator to go over the Jacobian array
        sum_s = np.sum(s[:i])
        if i == 0:
            gradient[i] += s[i] - z_k
        elif i == N:
            gradient[:i] += np.full(i, sum_s - z_k)
        else:
            gradient[i] += (
                ((y_k - c[i - 1]) * s[i] / delta_c + sum_s - z_k)
                * (y_k - c[i - 1])
                / delta_c
            )
            gradient[:i] += np.full(
                i,
                (y_k - c[i - 1]) * s[i] / delta_c + sum_s - z_k,
            )
    return gradient


def _calculate_regularization_for_group(
    s: np.ndarray,
    N: int,
    c: np.ndarray,
    regularization_factor: float,
):
    """Calculate regularization term the given spline.

    We regularize the spline to be linear. To do this, we calculate the optimal
    linear function that minimizes the sum of squared residuals with respect to
    the spline knots. Then we calculate the sum of squared residuals for this
    linear function. If the calculated offset is smaller than 0, we set it to 0.
    This is because the spline is not allowed to be negative.
    """
    # Calculate the spline knots xi_i from spline differences s_i
    lower_trian = np.tril(np.ones((N, N)))
    xi = np.dot(lower_trian, s)

    # Calculate auxiliary values
    c_sum = np.sum(c)
    xi_sum = np.sum(xi)
    c_squares_sum = np.sum(c**2)
    c_dot_xi = np.dot(c, xi)
    # Calculate the optimal linear function offset
    if np.isclose(N * c_squares_sum - c_sum**2, 0):
        beta_opt = xi_sum / N
    else:
        beta_opt = (xi_sum * c_squares_sum - c_dot_xi * c_sum) / (
            N * c_squares_sum - c_sum**2
        )

    # If the offset is smaller than 0, we set it to 0
    if beta_opt < 0:
        beta_opt = 0

    # Calculate the slope of the optimal linear function
    alpha_opt = (c_dot_xi - beta_opt * c_sum) / c_squares_sum

    # Calculate the sum of squared residuals for the optimal linear function
    regularization_term = np.sum((xi - alpha_opt * c - beta_opt) ** 2) / (
        2 * N
    )

    return regularization_term * regularization_factor


def _calculate_regularization_gradient_for_group(
    s: np.ndarray,
    N: int,
    c: np.ndarray,
    regularization_factor: float,
):
    """Calculate regularization term gradient for the given spline."""
    # Calculate the spline knots xi_i from spline differences s_i

    lower_trian = np.tril(np.ones((N, N)))
    xi = np.dot(lower_trian, s)

    # Calculate auxiliary values
    c_sum = np.sum(c)
    xi_sum = np.sum(xi)
    c_squares_sum = np.sum(c**2)
    c_dot_xi = np.dot(c, xi)

    # Calculate the optimal linear function offset
    if np.isclose(N * c_squares_sum - c_sum**2, 0):
        beta_opt = xi_sum / N
    else:
        beta_opt = (xi_sum * c_squares_sum - c_dot_xi * c_sum) / (
            N * c_squares_sum - c_sum**2
        )

    # If the offset is smaller than 0, we set it to 0.
    # Otherwise, we calculate the gradient of the offset.
    if beta_opt < 0:
        beta_opt = 0

    # Calculate the slope of the optimal linear function
    alpha_opt = (c_dot_xi - beta_opt * c_sum) / c_squares_sum

    # Calculate some more auxiliary values
    residuals = xi - alpha_opt * c - beta_opt

    # Can remove terms from this aux_matrix due to optimality
    # of the linear function (alpha & beta)
    aux_matrix = lower_trian

    # Calculate the gradient of the sum of squared residuals
    regularization_gradient = residuals @ aux_matrix / N

    return regularization_gradient * regularization_factor


def get_spline_mapped_simulations(
    s: np.ndarray,
    sim_all: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Return model simulations mapped using the approximation spline."""
    mapped_simulations = np.zeros(len(sim_all))
    lower_trian = np.tril(np.ones((N, N)))
    xi = np.dot(lower_trian, s)

    for y_k, n_k, index in zip(sim_all, n, range(len(sim_all)), strict=True):
        interval_index = n_k - 1
        if interval_index == 0 or interval_index == N:
            mapped_simulations[index] = xi[interval_index]
        else:
            mapped_simulations[index] = (y_k - c[interval_index - 1]) * (
                xi[interval_index] - xi[interval_index - 1]
            ) / delta_c + xi[interval_index - 1]

    return mapped_simulations


def calculate_inner_hessian(
    s: np.ndarray,
    sim_all: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Calculate the hessian of the objective function for the reformulated inner problem."""

    hessian = np.zeros((N, N))

    for y_k, sigma_k, n_k in zip(sim_all, sigma, n, strict=True):
        sum_s = 0
        i = n_k - 1  # just the iterator to go over the Hessian matrix
        for j in range(i):
            sum_s += s[j]

        hessian[i][i] += (1 / sigma_k**2) * ((y_k - c[i - 1]) / delta_c) ** 2
        for j in range(i):
            hessian[i][j] += (1 / sigma_k**2) * ((y_k - c[i - 1]) / delta_c)
            hessian[j][i] += (1 / sigma_k**2) * ((y_k - c[i - 1]) / delta_c)
            for h in range(i):
                hessian[j][h] += 1 / sigma_k**2

    return hessian


def calculate_dy_term(
    sim_all: np.ndarray,
    sy_all: np.ndarray,
    measurements: np.ndarray,
    s: np.ndarray,
    N: int,
    delta_c: float,
    delta_c_dot: float,
    c: np.ndarray,
    c_dot: np.ndarray,
    n: np.ndarray,
):
    """Calculate the derivative of the objective function for one group with respect to the simulations."""
    df_dy = 0

    for y_k, z_k, y_dot_k, n_k in zip(
        sim_all, measurements, sy_all, n, strict=True
    ):
        i = n_k - 1
        sum_s = np.sum(s[:i])
        if i > 0 and i < N:
            df_dy += (
                ((y_k - c[i - 1]) * s[i] / delta_c + sum_s - z_k)
                * s[i]
                * (
                    (y_dot_k - c_dot[i - 1]) * delta_c
                    - (y_k - c[i - 1]) * delta_c_dot
                )
                / delta_c**2
            )
        # There is no i==0 case, because in this case
        # c[0] == y_k and so the derivative is zero.
    return df_dy


def _beta_group_gradient(
    solver,
    sim_all: np.ndarray,
    sy_all: np.ndarray,
    measurements: np.ndarray,
    inner_pars: np.ndarray,
):
    """Per-group beta gradient contribution, with the domain guard, so both gradient paths call one line.

    A theta whose simulations cannot anchor a domain contributes nothing rather than raising: the outer
    optimizer already sees an infinite objective there from ``_optimize_beta``.
    """
    try:
        domain = solver._rescale_beta_domain(
            sim_all,
            solver.options[BETA_KAPPA],
            solver.options[BETA_BETA_STAR],
        )
    except ValueError:
        return np.zeros(sy_all.shape[0])
    return calculate_beta_dy_term(
        sim_all=sim_all,
        sy_all=sy_all,
        measurements=measurements,
        a=float(inner_pars[0]),
        b=float(inner_pars[1]),
        domain=domain,
        kappa=solver.options[BETA_KAPPA],
    )


def calculate_beta_dy_term(
    sim_all: np.ndarray,
    sy_all: np.ndarray,
    measurements: np.ndarray,
    a: float,
    b: float,
    domain: dict,
    kappa: float,
):
    """Gradient of one beta group's inner objective with respect to the outer parameters.

    Shared by both gradient paths, so the formula exists once. ``sy_all`` is ``(n_parameters,
    n_datapoints)``; the return is ``(n_parameters,)``.

    Offset, scale, sigma and the shapes are all at their inner optimum, so the envelope theorem leaves
    only the explicit dependence through ``x``. The anchors are NOT estimated, so they do contribute:

        dnllh/dtheta = (K/RSS) sum_i (ghat_i - y_i) s pdf(x_i) dx_i/dtheta
        dx_i/dtheta  = [dq_i - (1-x_i) dL - x_i dQ] / (Q - L)
        dbeta/dtheta = -beta * mean_i(q_i dq_i)/mean_i(q_i^2)
        dQ = kappa (w . dq + dlse_dbeta  dbeta) ,   dL = v . dq + dsmin_dbeta dbeta

    ``dx`` uses a single division: the quotient-rule form would square ``Q - L``, which gets as small as
    1e-10. No sigma appears because sigma is concentrated out into ``(K/2) log(RSS/K)``.
    """
    n_parameters = sy_all.shape[0]
    Q, L = domain["Q"], domain["L"]
    x = (sim_all - L) / (Q - L)
    K = len(measurements)

    with np.errstate(all="ignore"):
        z = betainc(a, b, x)
        log_pdf = (
            (a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - betaln(a, b)
        )
        pdf = np.exp(log_pdf)
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(pdf))):
        return np.zeros(n_parameters)

    y_centered = measurements - measurements.mean()
    sst = float(y_centered @ y_centered)
    z_centered = z - z.mean()
    d = float(z_centered @ z_centered)
    if d <= 1e-28 or sst <= 0.0:
        return np.zeros(n_parameters)
    n = float(z_centered @ y_centered)
    rss_raw = sst - n * n / d
    rss = _floor_rss(rss_raw, sst)
    if rss_raw < rss:
        # Floored: the objective is constant in theta there, so the derivative is zero.
        return np.zeros(n_parameters)

    scale = n / d
    residuals = (measurements.mean() + scale * (z - z.mean())) - measurements

    # mean_i(q dq)/mean_i(q^2): the 1/n_rows cancels, so this never forms rms explicitly.
    dbeta = -domain["beta"] * (sy_all @ sim_all) / float(sim_all @ sim_all)
    dQ = kappa * (sy_all @ domain["w"] + domain["dlse_dbeta"] * dbeta)
    dL = sy_all @ domain["v"] + domain["dsmin_dbeta"] * dbeta

    dx = (
        sy_all - np.outer(dL, 1.0 - x) - np.outer(dQ, x)
    ) / (Q - L)
    gradient = (K / rss) * scale * ((pdf * dx) @ residuals)
    return np.where(np.isfinite(gradient), gradient, 0.0)


def calculate_spline_bases_gradient(
    sim_all: np.ndarray, sy_all: np.ndarray, N: int
):
    """Calculate gradient of the rescaled spline bases."""

    min_idx = np.argmin(sim_all)
    max_idx = np.argmax(sim_all)

    min_all = sim_all[min_idx]
    max_all = sim_all[max_idx]
    # Coming directly from differentiating _rescale_spline_bases
    if sim_all[max_idx] - sim_all[min_idx] < MIN_SIM_RANGE:
        delta_c_dot = 0
        c_dot = np.full(N, (sy_all[max_idx] - sy_all[min_idx]) / 2)
        average_value = (max_all + min_all) / 2
        if average_value < (MIN_SIM_RANGE / 2):
            c_dot = np.full(N, 0)
        else:
            c_dot = np.full(N, (sy_all[max_idx] - sy_all[min_idx]) / 2)
    else:
        delta_c_dot = (sy_all[max_idx] - sy_all[min_idx]) / (N - 1)
        c_dot = np.linspace(sy_all[min_idx], sy_all[max_idx], N)

    return delta_c_dot, c_dot


def extract_expdata_using_mask(
    expdata: list[np.ndarray], mask: list[np.ndarray]
):
    """Extract data from expdata list of arrays for the given mask."""
    return np.concatenate(
        [
            expdata[condition_index][mask[condition_index]]
            for condition_index in range(len(mask))
        ]
    )


def save_inner_parameters_to_inner_problem(
    inner_problem: SemiquantProblem,
    s: np.ndarray,
    group: int,
) -> None:
    """Save inner parameter values to the inner subproblem.

    Calculates the non-reformulated inner spline parameters from
    the reformulated inner spline parameters and saves them to
    the inner subproblem.

    Parameters
    ----------
    inner_parameters : list
        List of inner parameters.
    s : np.ndarray
        Reformulated inner spline parameters.
    """
    group_dict = inner_problem.groups[group]
    inner_noise_parameters = inner_problem.get_noise_parameters_for_group(
        group
    )

    # `s` carries one entry per inner parameter of the group in BOTH families, regardless of which are
    # free: the spline solves in N_SPLINE_PARS dimensions and `_optimize_beta` always returns (a, b),
    # and neither consults the free/fixed split for its dimension. So this pairs with ALL xs.
    # strict=True turns any future divergence between the two into an error here rather than a silent
    # truncation or an IndexError further along.
    for inner_parameter, value in zip(
        inner_problem.get_xs_for_group(group), s, strict=True
    ):
        inner_parameter.value = value

    sigma = group_dict[INNER_NOISE_PARS]

    if group_dict[OPTIMIZE_NOISE]:
        inner_noise_parameters[0].value = sigma


def get_monotonicity_measure(measurement, simulation):
    """Get monotonicity measure by calculating inversions.

    Calculates the number of inversions in the simulation data
    with respect to the measurement data.

    Parameters
    ----------
    measurement : np.ndarray
        Measurement data.
    simulation : np.ndarray
        Simulation data.

    Returns
    -------
    inversions : int
        Number of inversions.
    """
    if len(measurement) != len(simulation):
        raise ValueError(
            "Measurement and simulation data must have the same length."
        )

    ordered_simulation = [
        x
        for _, x in sorted(
            zip(measurement, simulation, strict=True), key=lambda pair: pair[0]
        )
    ]
    ordered_measurement = sorted(simulation)

    inversions = 0
    for i in range(len(ordered_simulation)):
        for j in range(i + 1, len(ordered_simulation)):
            if ordered_simulation[i] > ordered_simulation[j]:
                inversions += 1
            elif (
                ordered_simulation[i] == ordered_simulation[j]
                and ordered_measurement[i] != ordered_measurement[j]
            ):
                inversions += 1

    return inversions
