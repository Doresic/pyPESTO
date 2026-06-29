"""Calculator for binary hierarchical optimization."""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence

import numpy as np

from ...C import (
    AMICI_SY,
    AMICI_Y,
    FVAL,
    GRAD,
    HESS,
    INNER_PARAMETERS,
    MODE_RES,
    RDATAS,
    RES,
    SRES,
)
from ...objective.amici.amici_calculator import (
    AmiciCalculator,
    AmiciModel,
    AmiciSolver,
)
from ...objective.amici.amici_util import (
    filter_return_dict,
    init_return_values,
)
from ..relative.calculator import get_sensitivities_from_adjoint_gradient
from .problem import BinaryInnerProblem
from .solver import BinaryInnerSolver

try:
    import amici

    from .. import _amici_patches  # noqa: F401  (must precede the import below)
    from amici.petab.conditions import fill_in_parameters
    from amici.petab.parameter_mapping import ParameterMapping
except ImportError:
    pass

logger = logging.getLogger(__name__)


class BinaryAmiciCalculator(AmiciCalculator):
    """Calculator for binary (Bernoulli) hierarchical data.

    Can be used as a standalone calculator (runs AMICI itself when
    ``rdatas`` is not provided) or as a component inside
    :class:`pypesto.hierarchical.inner_calculator_collector.InnerCalculatorCollector`
    which passes pre-computed ``rdatas``.

    Sensitivity notes
    -----------------
    - **Forward sensitivity**: use ``rdata.sy``.
    - **Adjoint sensitivity**: reconstruct ``sy`` from AMICI's likelihood
      gradient for Frohlich-style one-datapoint-per-condition problems.
    Both paths use :meth:`BinaryInnerSolver.calculate_gradients`.

    Parameters
    ----------
    inner_problem:
        The binary inner problem constructed from PEtab data.
    inner_solver:
        Solver for the inner logistic regression.  Defaults to a new
        :class:`BinaryInnerSolver`.
    use_firth:
        Forwarded to a default :class:`BinaryInnerSolver` (ignored if
        ``inner_solver`` is given).
    binary_link:
        Link function (``"logit"`` default, or ``"probit"``) forwarded to a
        default :class:`BinaryInnerSolver` (ignored if ``inner_solver`` is
        given).
    """

    def __init__(
        self,
        inner_problem: BinaryInnerProblem,
        inner_solver: BinaryInnerSolver | None = None,
        use_firth: bool = True,
        binary_link: str = "logit",
    ):
        super().__init__()
        self.inner_problem = inner_problem
        if inner_solver is None:
            inner_solver = BinaryInnerSolver(
                use_firth=use_firth, binary_link=binary_link
            )
        self.inner_solver = inner_solver
        self._recalc_plists_and_scales = True

    def initialize(self) -> None:
        """Re-initialize inner problem and solver."""
        super().initialize()
        self.inner_problem.initialize()
        self.inner_solver.initialize()

    def __call__(
        self,
        x_dct: dict,
        sensi_orders: tuple[int, ...],
        mode: str,
        amici_model: AmiciModel,
        amici_solver: AmiciSolver,
        edatas: list,
        n_threads: int,
        x_ids: Sequence[str],
        parameter_mapping: "ParameterMapping",
        fim_for_hess: bool,
        rdatas: list | None = None,
    ) -> dict:
        """Evaluate the binary likelihood and (optionally) its gradient.

        Parameters
        ----------
        x_dct:
            Current outer parameter values (dict: parameter_id → value).
        sensi_orders:
            Tuple of requested sensitivity orders (0 for function value,
            1 for gradient).
        mode:
            Objective mode.  ``MODE_RES`` is not supported for binary data.
        amici_model:
            The compiled AMICI model.
        amici_solver:
            The AMICI solver (configured externally).
        edatas:
            AMICI ``ExpData`` objects.
        n_threads:
            Number of parallel AMICI threads.
        x_ids:
            Outer (optimization) parameter IDs — defines the gradient
            vector dimension.
        parameter_mapping:
            PEtab → AMICI parameter mapping.
        fim_for_hess:
            Ignored (Hessian not implemented for binary data).
        rdatas:
            Pre-computed AMICI return data.  If ``None``, the model is
            simulated here.  Pass this from
            ``InnerCalculatorCollector`` to avoid redundant simulations.

        Returns
        -------
        dict
            Keys: ``FVAL``, ``GRAD``, ``HESS``, ``RES``, ``SRES``,
            ``RDATAS``, ``INNER_PARAMETERS``.
        """
        if not self.inner_problem.check_edatas(edatas=edatas):
            raise ValueError(
                "The experimental data provided to this call differs from "
                "the experimental data used to setup the binary hierarchical "
                "optimizer."
            )

        if mode == MODE_RES:
            raise ValueError(
                "BinaryAmiciCalculator does not support residual mode."
            )
        if 2 in sensi_orders:
            raise ValueError(
                "Hessian is not implemented for BinaryAmiciCalculator."
            )

        dim = len(x_ids)
        nllh, snllh, s2nllh, chi2, res, sres = init_return_values(
            sensi_orders, mode, dim
        )
        sensi_order = max(sensi_orders) if sensi_orders else 0

        # --- run AMICI if not provided ---
        if rdatas is None:
            amici_solver.setSensitivityOrder(sensi_order)
            x_dct = copy.deepcopy(x_dct)
            fill_in_parameters(
                edatas=edatas,
                problem_parameters=x_dct,
                scaled_parameters=True,
                parameter_mapping=parameter_mapping,
                amici_model=amici_model,
                recalc_plists_and_scales=self._recalc_plists_and_scales,
            )
            if self._recalc_plists_and_scales:
                self._recalc_plists_and_scales = False

            rdatas = amici.runAmiciSimulations(
                amici_model,
                amici_solver,
                edatas,
                num_threads=min(n_threads, len(edatas)),
            )

        inner_result = {
            FVAL: nllh,
            GRAD: snllh,
            HESS: s2nllh,
            RES: res,
            SRES: sres,
            RDATAS: rdatas,
        }

        # fail fast on AMICI errors
        if any(rdata.status != amici.AMICI_SUCCESS for rdata in rdatas):
            inner_result[FVAL] = np.inf
            if 1 in sensi_orders:
                inner_result[GRAD] = np.full(dim, np.nan)
            return filter_return_dict(inner_result)

        # extract sim: list of rdata.y arrays (one per condition)
        sim = [rdata[AMICI_Y] for rdata in rdatas]

        # --- inner optimization ---
        x_inner = self.inner_solver.solve(self.inner_problem, sim)
        nllh = self.inner_solver.calculate_nllh(self.inner_problem, sim, x_inner)

        inner_result[FVAL] = nllh
        inner_result[INNER_PARAMETERS] = np.array(
            [
                self.inner_problem.xs[xid].value
                for xid in self.inner_problem.get_x_ids()
            ]
        )

        # --- outer gradient ---
        if sensi_order > 0:
            if (
                amici_solver.getSensitivityMethod()
                == amici.SensitivityMethod_forward
            ):
                sy = [rdata[AMICI_SY] for rdata in rdatas]
            elif (
                amici_solver.getSensitivityMethod()
                == amici.SensitivityMethod_adjoint
            ):
                y = self.inner_solver._extract_y(self.inner_problem, sim)
                n_equal_label = int(
                    np.sum(np.abs(y - self.inner_problem.labels) < 1e-12)
                )
                if n_equal_label:
                    logger.warning(
                        "Binary adjoint sensitivity reconstruction drops "
                        "gradient contributions for %d measurements where "
                        "simulated y equals the binary label. Use forward "
                        "sensitivities to check those conditions.",
                        n_equal_label,
                    )
                sy = list(get_sensitivities_from_adjoint_gradient(rdatas, edatas))
            else:
                raise ValueError(
                    "Unsupported AMICI sensitivity method for binary "
                    "hierarchical gradients."
                )

            snllh = self.inner_solver.calculate_gradients(
                problem=self.inner_problem,
                sim=sim,
                sy=sy,
                x_inner=x_inner,
                parameter_mapping=parameter_mapping,
                par_opt_ids=list(x_ids),
                par_sim_ids=list(amici_model.getParameterIds()),
                par_edatas_indices=[edata.plist for edata in edatas],
                snllh=snllh,
            )
            inner_result[GRAD] = snllh

        return filter_return_dict(inner_result)
