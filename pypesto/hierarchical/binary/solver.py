"""Inner solver for binary hierarchical optimization."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # σ(x) = 1 / (1 + exp(-x))

from .problem import BinaryInnerProblem

logger = logging.getLogger(__name__)


class BinaryInnerSolver:
    """Solves the binary inner optimization problem for fixed outer parameters.

    The inner problem is logistic regression (convex), solved with
    ``scipy.optimize.minimize`` using the ``L-BFGS-B`` method and
    analytic gradients.

    Inner parameter layout
    ----------------------
    ``x_inner = [α_0, α_1, …, α_{G-1}, β]``  (length G + 1)

    The outer gradient is computed via the envelope theorem:

    .. math::
        \\frac{\\partial(-\\ell)}{\\partial \\theta_k}
        = -\\sum_i (z_i - p^*_i)\\, \\beta^*\\, sy_{i,k}

    where :math:`sy_{i,k} = \\partial y_i / \\partial \\theta_k` is taken
    directly from AMICI forward sensitivities.
    """

    def solve(
        self,
        problem: BinaryInnerProblem,
        sim: list[np.ndarray],
    ) -> np.ndarray:
        """Optimize α and β for the current outer parameters.

        Parameters
        ----------
        problem:
            The binary inner problem.
        sim:
            ``rdata.y`` for each simulation condition (list of arrays,
            each of shape ``(n_timepoints, n_observables)``).

        Returns
        -------
        x_inner: np.ndarray, shape ``(n_alpha + 1,)``
            Optimal ``[α_0, …, α_{G-1}, β]``.
        """
        y = self._extract_y(problem, sim)  # shape (n_meas,)
        z = problem.labels
        ag = problem.alpha_group_ixs
        n_alpha = problem.n_alpha

        lb, ub = problem.get_bounds()
        scipy_bounds = [
            (
                None if np.isinf(lb[k]) else lb[k],
                None if np.isinf(ub[k]) else ub[k],
            )
            for k in range(n_alpha + 1)
        ]

        x0 = np.zeros(n_alpha + 1)

        def _nll_and_grad(x_inner: np.ndarray) -> tuple[float, np.ndarray]:
            alpha = x_inner[:n_alpha]
            beta = x_inner[-1]
            lin_pred = alpha[ag] + beta * y
            p = expit(lin_pred)
            p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
            nll = -np.sum(z * np.log(p_clip) + (1.0 - z) * np.log(1.0 - p_clip))
            residual = p - z  # shape (n_meas,)
            grad_beta = float(np.dot(residual, y))
            grad_alpha = np.zeros(n_alpha)
            np.add.at(grad_alpha, ag, residual)
            return nll, np.append(grad_alpha, grad_beta)

        result = minimize(
            _nll_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=scipy_bounds,
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )

        if not result.success:
            logger.warning(
                "Binary inner optimization did not converge: %s", result.message
            )

        # store optimal values back into parameter objects
        x_opt = result.x
        for k, aid in enumerate(problem.alpha_ids):
            problem.xs[aid].value = float(x_opt[k])
        problem.xs[problem.beta_id].value = float(x_opt[-1])

        return x_opt

    def calculate_nllh(
        self,
        problem: BinaryInnerProblem,
        sim: list[np.ndarray],
        x_inner: np.ndarray,
    ) -> float:
        """Evaluate the negative log-likelihood at ``x_inner``.

        Parameters
        ----------
        problem:
            The binary inner problem.
        sim:
            ``rdata.y`` per condition.
        x_inner:
            Inner parameter vector ``[α_0, …, α_{G-1}, β]``.

        Returns
        -------
        float
            Negative log-likelihood ``−ℓ``.
        """
        y = self._extract_y(problem, sim)
        z = problem.labels
        alpha = x_inner[: problem.n_alpha]
        beta = x_inner[-1]
        lin_pred = alpha[problem.alpha_group_ixs] + beta * y
        p = expit(lin_pred)
        p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
        return float(
            -np.sum(z * np.log(p_clip) + (1.0 - z) * np.log(1.0 - p_clip))
        )

    def calculate_gradients(
        self,
        problem: BinaryInnerProblem,
        sim: list[np.ndarray],
        sy: list[np.ndarray | None],
        x_inner: np.ndarray,
        par_opt_ids: list[str],
        par_sim_ids: list[str],
    ) -> np.ndarray:
        """Compute the outer gradient ``∂(−ℓ)/∂θ``.

        Uses the envelope theorem: at the optimal inner parameters
        ``(α*, β*)``, the total derivative equals the partial derivative,
        so

        .. math::
            \\frac{\\partial(-\\ell)}{\\partial \\theta_k}
            = -\\sum_i (z_i - p^*_i)\\, \\beta^*\\, sy_{i,k}

        Requires AMICI forward sensitivities: ``rdata.sy`` shape
        ``(n_timepoints, n_par_sim, n_observables)``.

        If any ``sy`` entry is ``None`` (e.g. adjoint run), returns
        ``np.zeros(n_par_opt)``.

        Parameters
        ----------
        problem:
            The binary inner problem.
        sim:
            ``rdata.y`` per condition (list, each ``(n_t, n_obs)``).
        sy:
            ``rdata.sy`` per condition (list, each
            ``(n_t, n_par_sim, n_obs)`` or ``None``).
        x_inner:
            Optimal inner parameters ``[α_0, …, α_{G-1}, β]``.
        par_opt_ids:
            Outer (optimization) parameter IDs — defines the output vector.
        par_sim_ids:
            AMICI simulation parameter IDs — indexes ``rdata.sy``'s
            second axis.

        Returns
        -------
        snllh: np.ndarray, shape ``(n_par_opt,)``
            Gradient of the negative log-likelihood w.r.t. outer params.
        """
        n_par_opt = len(par_opt_ids)
        snllh = np.zeros(n_par_opt)

        if any(s is None for s in sy):
            logger.debug(
                "Binary gradient skipped: sy is None (adjoint sensitivity run)."
            )
            return snllh

        y = self._extract_y(problem, sim)
        alpha = x_inner[: problem.n_alpha]
        beta = x_inner[-1]
        lin_pred = alpha[problem.alpha_group_ixs] + beta * y
        p = expit(lin_pred)
        residual = p - problem.labels  # (z - p) negated → gives ∂(−ℓ)/∂θ sign

        # build sim-to-opt index mapping
        par_opt_set = {pid: i for i, pid in enumerate(par_opt_ids)}
        sim_to_opt: dict[int, int] = {}
        for sim_ix, sim_id in enumerate(par_sim_ids):
            if sim_id in par_opt_set:
                sim_to_opt[sim_ix] = par_opt_set[sim_id]

        if not sim_to_opt:
            logger.warning(
                "Binary gradient: no overlap between par_sim_ids and "
                "par_opt_ids. Returning zero gradient."
            )
            return snllh

        # accumulate gradient
        for i in range(problem.n_meas):
            ci = problem.cond_ixs[i]
            ti = problem.time_ixs[i]
            oi = problem.obs_ixs[i]
            # sy[ci] shape: (n_t, n_par_sim, n_obs)
            sy_i = sy[ci][ti, :, oi]  # shape (n_par_sim,)
            scale = residual[i] * beta
            for sim_ix, opt_ix in sim_to_opt.items():
                snllh[opt_ix] += scale * sy_i[sim_ix]

        return snllh

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_y(
        problem: BinaryInnerProblem,
        sim: list[np.ndarray],
    ) -> np.ndarray:
        """Extract the model observable values for all binary measurements.

        Parameters
        ----------
        problem:
            The binary inner problem (provides index arrays).
        sim:
            ``rdata.y`` per condition, each ``(n_t, n_obs)``.

        Returns
        -------
        y: np.ndarray, shape ``(n_meas,)``
            Observable value for each binary measurement.
        """
        return np.array(
            [
                sim[problem.cond_ixs[i]][problem.time_ixs[i], problem.obs_ixs[i]]
                for i in range(problem.n_meas)
            ],
            dtype=float,
        )
