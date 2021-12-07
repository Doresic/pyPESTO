from typing import Dict, List, Sequence, Union
import copy
import numpy as np

from ..objective.amici_calculator import (
    AmiciCalculator, calculate_function_values)
from ..objective.amici_util import (
    get_error_output)
from ..objective.constants import FVAL, GRAD, HESS, RES, SRES, RDATAS
from .problem import InnerProblem
from .optimal_scaling_problem import OptimalScalingProblem
from .solver import InnerSolver, AnalyticalInnerSolver
from .optimal_scaling_solver import OptimalScalingInnerSolver

try:
    import amici
    from amici.parameter_mapping import ParameterMapping
except ImportError:
    amici = None

AmiciModel = Union['amici.Model', 'amici.ModelPtr']
AmiciSolver = Union['amici.Solver', 'amici.SolverPtr']


class HierarchicalAmiciCalculator(AmiciCalculator):
    """
    A calculator is passed as `calculator` to the pypesto.AmiciObjective.
    While this class cannot be used directly, it has two subclasses
    which allow to use forward or adjoint sensitivity analysis to
    solve a `pypesto.HierarchicalProblem` efficiently in an inner loop,
    while the outer optimization is only concerned with variables not
    specified as `pypesto.HierarchicalParameter`s.
    """

    def __init__(self,
                 inner_problem: OptimalScalingProblem,
                 inner_solver: InnerSolver = None):
        """
        Initialize the calculator from the given problem.
        """
        super().__init__()

        self.inner_problem = inner_problem

        if inner_solver is None:
            # inner_solver = NumericalInnerSolver()
            # inner_solver = AnalyticalInnerSolver()
            inner_solver = OptimalScalingInnerSolver()
        self.inner_solver = inner_solver

    def initialize(self):
        super().initialize()
        self.inner_solver.initialize()

    def __call__(self,
                 x_dct: Dict,
                 sensi_order: int,
                 mode: str,
                 amici_model: AmiciModel,
                 amici_solver: AmiciSolver,
                 edatas: List['amici.ExpData'],
                 n_threads: int,
                 x_ids: Sequence[str],
                 parameter_mapping: 'ParameterMapping',
                 fim_for_hess: bool):

        dim = len(x_ids)
        nllh = 0.0
        snllh = np.zeros(dim)
        s2nllh = np.zeros([dim, dim])

        res = np.zeros([0])
        sres = np.zeros([0, dim])

        # set order in solver to 0
        # amici_solver.setSensitivityOrder(0)
        amici_solver.setSensitivityOrder(sensi_order)

        # fill in boring values
        x_dct = copy.deepcopy(x_dct)
        #print(x_dct)
        for key, val in self.inner_problem.get_boring_pars(
                scaled=True).items():
            x_dct[key] = val

        # fill in parameters
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

        # check if any simulation failed
        if any([rdata['status'] < 0.0 for rdata in rdatas]):
            return get_error_output(amici_model, edatas, rdatas, sensi_order, mode, dim)

        sim = [rdata['y'] for rdata in rdatas]
        sigma = [rdata['sigmay'] for rdata in rdatas]

        # compute optimal inner parameters

        x_inner_opt = self.inner_solver.solve(
            self.inner_problem, sim, sigma, scaled=True)

        nllh = self.inner_solver.calculate_obj_function(x_inner_opt)

        # print(x_inner_opt)

        # if sensi_order == 0:
        #     dim = len(x_ids)
        #     nllh = compute_nllh(self.inner_problem.data, sim, sigma)
        #     return {
        #         FVAL: nllh,
        #         GRAD: np.zeros(dim),
        #         HESS: np.zeros([dim, dim]),
        #         RES: np.zeros([0]),
        #         SRES: np.zeros([0, dim]),
        #         RDATAS: rdatas
        #     }

        # fill in optimal values
        # TODO: x_inner_opt is different for hierarchical and
        #  qualitative approach. For now I commented the following
        #  lines out to make qualitative approach work.
        # x_dct = copy.deepcopy(x_dct)
        # for key, val in x_inner_opt.items():
        #    x_dct[key] = val

        # fill in parameters
        # TODO (#226) use plist to compute only required derivatives
        amici.parameter_mapping.fill_in_parameters(
            edatas=edatas,
            problem_parameters=x_dct,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=amici_model
        )

        if sensi_order > 0:
            #amici_solver.setSensitivityOrder(sensi_order)
            num_model_pars = len(amici_model.getParameterIds())
            # resimulate
            # run amici simulation
            #rdatas = amici.runAmiciSimulations(
            #    amici_model,
            #    amici_solver,
            #    edatas,
            #    num_threads=min(n_threads, len(edatas)),
            #)
            sy = [rdata['sy'] for rdata in rdatas]
            #print("Evo sy: \n", sy)
            snllh = self.inner_solver.calculate_gradients(self.inner_problem,
                                                          x_inner_opt,
                                                          sim,
                                                          sy,
                                                          parameter_mapping,
                                                          x_ids,
                                                          amici_model,
                                                          snllh)
            #print("Evo snllh: \n", snllh)

        return {FVAL: nllh,
                GRAD: snllh,
                HESS: s2nllh,
                RES: res,
                SRES: sres,
                RDATAS: rdatas
                }

        #return calculate_function_values(
        #    rdatas, sensi_order, mode, amici_model, amici_solver, edatas,
        #    x_ids, parameter_mapping,fim_for_hess)
