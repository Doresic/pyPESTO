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

        # x_dct['Epo_degradation_BaF3'] = -1.568917588
        # x_dct['k_exp_hetero'] = -5
        # x_dct['k_exp_homo'] = -2.209698782
        # x_dct['k_imp_hetero'] = -1.786006548
        # x_dct['k_imp_homo'] = 4.990114009
        # x_dct['k_phos'] = 4.197735489


        # x_dct['K_1'] = -4.999994697
        # x_dct['K_2'] = 0.179327683
        # x_dct['K_3'] = -0.01751816
        # x_dct['k10'] = -1.578171005
        # x_dct['k11'] = 2.212619999
        # x_dct['k2'] = 0.088569897
        # x_dct['k3'] = 2.886016418
        # x_dct['k4'] = 0.463435212
        # x_dct['k5'] = 2.761591228
        # x_dct['k6'] = 0.557215773
        # x_dct['tau1'] = -0.614711949
        # x_dct['tau2'] = -1.674875963
        
        # x_dct['CD274mRNA_production'] = -1.673482804
        # x_dct['DecoyR_binding'] = -2.414422068
        # x_dct['JAK2_p_inhibition'] = -1.10712659
        # x_dct['JAK2_phosphorylation'] = 0.007411355
        # x_dct['Kon_IL13Rec'] = -2.642419383
        # x_dct['Rec_intern'] = -0.463278373
        # x_dct['Rec_phosphorylation'] = 2.99999992
        # x_dct['Rec_recycle'] = -2.678246463
        # x_dct['SOCS3_accumulation'] = 2.654924984
        # x_dct['SOCS3_degradation'] = -1.362708216
        # x_dct['SOCS3_translation'] = 1.196416317
        # x_dct['SOCS3mRNA_production'] = -0.843537117
        # x_dct['STAT5_phosphorylation'] = -1.710364794
        # x_dct['init_Rec_i'] = 2.371175949
        # x_dct['pJAK2_dephosphorylation'] = -3.754748951
        # x_dct['pRec_degradation'] = -0.678662642
        # x_dct['pRec_intern'] = -0.232642597
        # x_dct['pSTAT5_dephosphorylation'] = -3.560932197


        #print(x_dct)
        #breakpoint()
        
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
        #print(sim)
        #Sometimes some simulations are very small negative numbers, so:
        for i in range(len(sim)):
            sim[i]=sim[i].clip(min=0)

        sigma = [rdata['sigmay'] for rdata in rdatas]

        # compute optimal inner parameters

        x_inner_opt = self.inner_solver.solve(
            self.inner_problem, sim, sigma, scaled=True)

        #if(self.inner_solver.options['InnerOptimizer']=='SLSQP'):
        nllh = self.inner_solver.calculate_obj_function(x_inner_opt)
        # elif(self.inner_solver.options['InnerOptimizer']=='LeastSquares'):
        #     nllh= self.inner_solver.ls_calculate_obj_function(x_inner_opt)
        # else:
        #     print("Wrong choice of inner optimizer")
        #     breakpoint()

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
            

            #if(self.inner_solver.options['InnerOptimizer']=='SLSQP'):
            snllh = self.inner_solver.calculate_gradients(self.inner_problem,
                                                              x_inner_opt,
                                                              sim,
                                                              sy,
                                                              parameter_mapping,
                                                              x_ids,
                                                              amici_model,
                                                              snllh,
                                                              sigma)
            # elif(self.inner_solver.options['InnerOptimizer']=='LeastSquares'):
            #     snllh = self.inner_solver.ls_calculate_gradients(self.inner_problem,
            #                                                         x_inner_opt,
            #                                                         sim,
            #                                                         sy,
            #                                                         parameter_mapping,
            #                                                         x_ids,
            #                                                         amici_model,
            #                                                         snllh,
            #                                                         sigma)
            # else:
            #     print("Wrong choice of inner optimizer")
            #     breakpoint()
            
    

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
