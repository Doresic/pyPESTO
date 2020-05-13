import numpy as np
from typing import Dict, List, Sequence, Union

from .constants import (
    MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, RDATAS, CHI2
)
from .amici_util import (
    add_sim_grad_to_opt_grad, add_sim_hess_to_opt_hess,
    sim_sres_to_opt_sres, log_simulation, get_error_output)

try:
    import amici
    import amici.petab_objective
    import amici.parameter_mapping
    from amici.parameter_mapping import ParameterMapping
except ImportError:
    pass

AmiciModel = Union['amici.Model', 'amici.ModelPtr']
AmiciSolver = Union['amici.Solver', 'amici.SolverPtr']


class AmiciCalculator:
    """
    Class to perform the actual call to AMICI and obtain requested objective
    function values.
    """

    def __init__(self):
        self._known_least_squares_safe = False

    def initialize(self):
        """Initialize the calculator. Default: Do nothing."""

    def __call__(self,
                 x_dct: Dict,
                 sensi_order: int,
                 mode: str,
                 amici_model: AmiciModel,
                 amici_solver: AmiciSolver,
                 edatas: List['amici.ExpData'],
                 n_threads: int,
                 x_ids: Sequence[str],
                 parameter_mapping: 'ParameterMapping'):
        """Perform the actual AMICI call.

        This function is called inside :func:`pypesto.AmiciObjective.__call__`
        after some preprocessing,
        and is supposed to return the function value, derivatives and
        possibly residuals as a dict for the given input.

        Parameters
        ----------
        x_dct:
            Parameters for which to compute function value and derivatives.
        sensi_order:
            Maximum sensitivity order.
        mode:
            Call mode (function value or residual based).
        amici_model:
            The AMICI model.
        amici_solver:
            The AMICI solver.
        edatas:
            The experimental data.
        n_threads:
            Number of threads for AMICI call.
        x_ids:
            Ids of optimization parameters.
        parameter_mapping:
            Mapping of optimization to simulation parameters.
        """
        # set order in solver
        amici_solver.setSensitivityOrder(sensi_order)

        # fill in parameters
        # TODO (#226) use plist to compute only required derivatives
        amici.parameter_mapping.fill_in_parameters(
            edatas=edatas,
            problem_parameters=x_dct,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=amici_model
        )

        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            amici_model,
            amici_solver,
            edatas,
            num_threads=min(n_threads, len(edatas)),
        )

        self._check_least_squares(sensi_order, mode, rdatas)

        return calculate_function_values(
            rdatas, sensi_order, mode, amici_model, amici_solver, edatas,
            x_ids, parameter_mapping)

    def _check_least_squares(
            self, sensi_order: int, mode: str, rdatas: List['amici.ExpData']):
        if not self._known_least_squares_safe and sensi_order > 0 \
                and mode == MODE_RES:
            if any(
                ((r['ssigmay'] is not None and np.any(r['ssigmay']))
                 or
                 (r['ssigmaz'] is not None and np.any(r['ssigmaz'])))
                for r in rdatas
            ):
                raise RuntimeError('Cannot use least squares solver with'
                                   'parameter dependent sigma!')
            self._known_least_squares_safe = True  # don't check this again


def calculate_function_values(rdatas,
                              sensi_order: int,
                              mode: str,
                              amici_model: AmiciModel,
                              amici_solver: AmiciSolver,
                              edatas: List['amici.ExpData'],
                              x_ids: Sequence[str],
                              parameter_mapping: 'ParameterMapping'):
    # full optimization problem dimension (including fixed parameters)
    dim = len(x_ids)

    # check if the simulation failed
    if any(rdata['status'] < 0.0 for rdata in rdatas):
        return get_error_output(amici_model, edatas, rdatas, dim)

    # prepare outputs
    nllh = 0.0
    snllh = None
    s2nllh = None
    if mode == MODE_FUN and sensi_order > 0:
        snllh = np.zeros(dim)
        s2nllh = np.zeros([dim, dim])

    chi2 = None
    res = None
    sres = None
    if mode == MODE_RES:
        chi2 = 0.0
        res = np.zeros([0])
        if sensi_order > 0:
            sres = np.zeros([0, dim])

    par_sim_ids = list(amici_model.getParameterIds())
    sensi_method = amici_solver.getSensitivityMethod()

    for data_ix, rdata in enumerate(rdatas):
        log_simulation(data_ix, rdata)

        condition_map_sim_var = \
            parameter_mapping[data_ix].map_sim_var

        nllh -= rdata['llh']

        # compute objective
        if mode == MODE_FUN:

            if sensi_order > 0:
                add_sim_grad_to_opt_grad(
                    x_ids,
                    par_sim_ids,
                    condition_map_sim_var,
                    rdata['sllh'],
                    snllh,
                    coefficient=-1.0
                )
                if sensi_method == 1:
                    # TODO Compute the full Hessian, and check here
                    add_sim_hess_to_opt_hess(
                        x_ids,
                        par_sim_ids,
                        condition_map_sim_var,
                        rdata['FIM'],
                        s2nllh,
                        coefficient=+1.0
                    )

        elif mode == MODE_RES:
            chi2 += rdata['chi2']
            res = np.hstack([res, rdata['res']]) \
                if res.size else rdata['res']
            if sensi_order > 0:
                opt_sres = sim_sres_to_opt_sres(
                    x_ids,
                    par_sim_ids,
                    condition_map_sim_var,
                    rdata['sres'],
                    coefficient=1.0
                )
                sres = np.vstack([sres, opt_sres]) \
                    if sres.size else opt_sres

    ret = {
        FVAL: nllh,
        CHI2: chi2,
        GRAD: snllh,
        HESS: s2nllh,
        RES: res,
        SRES: sres,
        RDATAS: rdatas
    }

    return {
        key: val
        for key, val in ret.items()
        if val is not None
    }
