"""Inner solver for binary hierarchical optimization."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # σ(x) = 1 / (1 + exp(-x))

from ..base_solver import InnerSolver
from ...objective.amici.amici_util import add_sim_grad_to_opt_grad
from .problem import BinaryInnerProblem

try:
    from amici.petab.parameter_mapping import ParameterMapping
except ImportError:
    pass

logger = logging.getLogger(__name__)

LARGE_INNER_PARAMETER_WARNING_THRESHOLD = 50.0


class BinaryInnerSolver(InnerSolver):
    """Solves the binary inner optimization problem for fixed outer parameters.

    The inner problem is logistic regression (convex), solved with
    ``scipy.optimize.minimize`` using the ``L-BFGS-B`` method and
    analytic gradients.

    Inner parameter layout
    ----------------------
    ``x_inner = [α_0, …, α_{G-1}, β_0, …, β_{B-1}]``
    (length G + B)

    The outer gradient is computed via the envelope theorem:

    .. math::
        \\frac{\\partial(-\\ell)}{\\partial \\theta_k}
        = -\\sum_i (z_i - p^*_i)\\, \\beta^*\\, sy_{i,k}

    where :math:`sy_{i,k} = \\partial y_i / \\partial \\theta_k` is taken
    directly from AMICI forward sensitivities.
    """

    def __init__(self, use_firth: bool = True):
        """Construct the binary inner solver.

        Parameters
        ----------
        use_firth:
            If True (default), penalise the inner logistic with Firth's
            Jeffreys-prior term ``+½ log|I|`` (``I = XᵀWX``). This makes the
            inner optimum finite and well-conditioned under (quasi-)complete
            separation, so the inner solve converges and the envelope-theorem
            outer gradient is valid. ``False`` recovers the plain unpenalised
            MLE (which can diverge under separation).
        """
        super().__init__()
        self.use_firth = use_firth

    def initialize(self) -> None:
        """Initialize the solver."""
        super().initialize()

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
        x_inner: np.ndarray, shape ``(n_alpha + n_beta,)``
            Optimal ``[α_0, …, α_{G-1}, β_0, …, β_{B-1}]``.
        """
        y = self._extract_y(problem, sim)  # shape (n_meas,)
        z = problem.labels
        n_inner = problem.n_alpha + problem.n_beta

        # Logistic design matrix: alpha-group indicator columns + beta-group
        # columns holding y_i. eta = X @ x_inner == alpha_{g(i)} + beta_{b(i)}*y_i.
        X = self._design_matrix(problem, y)

        lb, ub = problem.get_bounds()
        scipy_bounds = [
            (
                None if np.isinf(lb[k]) else lb[k],
                None if np.isinf(ub[k]) else ub[k],
            )
            for k in range(n_inner)
        ]

        x0 = np.zeros(n_inner)

        def _nll_and_grad(x_inner: np.ndarray) -> tuple[float, np.ndarray]:
            p = expit(X @ x_inner)
            p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
            nll = -np.sum(z * np.log(p_clip) + (1.0 - z) * np.log(1.0 - p_clip))
            resid_eff = p - z  # gradient weight per measurement
            if self.use_firth:
                # Firth: J = NLL - ½log|I|; score gains the -h(½-p) correction,
                # h_i = w_i q_i the (weighted) leverages. Reparam-invariant penalty.
                w = p * (1.0 - p)
                _, _, q, logdet = self._firth_info(X, w)
                nll = nll - 0.5 * logdet
                resid_eff = resid_eff - w * q * (0.5 - p)
            return nll, X.T @ resid_eff

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
        max_abs_x = np.max(np.abs(x_opt)) if x_opt.size else 0.0
        if max_abs_x > LARGE_INNER_PARAMETER_WARNING_THRESHOLD:
            if self.use_firth:
                logger.warning(
                    "Binary inner optimization returned large alpha/beta values "
                    "(max abs %.3g) despite Firth penalization — check for "
                    "extreme simulated viabilities or numerical issues.",
                    max_abs_x,
                )
            else:
                logger.warning(
                    "Binary inner optimization returned large alpha/beta values "
                    "(max abs %.3g). This may indicate complete or quasi-complete "
                    "separation in the unregularized logistic inner problem; "
                    "consider enabling Firth (use_firth=True).",
                    max_abs_x,
                )

        for k, aid in enumerate(problem.alpha_ids):
            problem.xs[aid].value = float(x_opt[k])
        for k, bid in enumerate(problem.beta_ids):
            problem.xs[bid].value = float(x_opt[problem.n_alpha + k])

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
            Inner parameter vector ``[α_0, …, α_{G-1}, β_0, …, β_{B-1}]``.

        Returns
        -------
        float
            Negative log-likelihood ``−ℓ``.
        """
        y = self._extract_y(problem, sim)
        z = problem.labels
        X = self._design_matrix(problem, y)
        p = expit(X @ x_inner)
        p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
        nllh = -np.sum(z * np.log(p_clip) + (1.0 - z) * np.log(1.0 - p_clip))
        if self.use_firth:
            # Outer FVAL must be the penalised objective so the envelope theorem
            # gives dF/dθ = ∂(NLL - ½log|I|)/∂θ cleanly.
            w = p * (1.0 - p)
            _, _, _, logdet = self._firth_info(X, w)
            nllh = nllh - 0.5 * logdet
        return float(nllh)

    def calculate_gradients(
        self,
        problem: BinaryInnerProblem,
        sim: list[np.ndarray],
        sy: list[np.ndarray | None],
        x_inner: np.ndarray,
        parameter_mapping: "ParameterMapping",
        par_opt_ids: list[str],
        par_sim_ids: list[str],
        par_edatas_indices: list[list[int]],
        snllh: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the outer gradient ``∂(−ℓ)/∂θ``.

        Uses the envelope theorem: at the optimal inner parameters
        ``(α*, β*)``, the total derivative equals the partial derivative,
        so

        .. math::
            \\frac{\\partial(-\\ell)}{\\partial \\theta_k}
            = \\sum_i (p_i^* - z_i)\\, \\beta^*_{b_i}\\, sy_{i,k}

        Requires AMICI forward sensitivities: ``rdata.sy`` shape
        ``(n_timepoints, n_par_sim, n_observables)``.

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
            Optimal inner parameters ``[α_0, …, α_{G-1}, β_0, …, β_{B-1}]``.
        parameter_mapping:
            PEtab to AMICI parameter mapping.
        par_opt_ids:
            Outer (optimization) parameter IDs — defines the output vector.
        par_sim_ids:
            AMICI simulation parameter IDs.
        par_edatas_indices:
            Per-condition AMICI plists; maps model simulation parameter
            indices to the parameter axis in ``sy``.
        snllh:
            Optional gradient vector to accumulate into.

        Returns
        -------
        snllh: np.ndarray, shape ``(n_par_opt,)``
            Gradient of the negative log-likelihood w.r.t. outer params.
        """
        if snllh is None:
            snllh = np.zeros(len(par_opt_ids))

        if any(s is None for s in sy):
            raise ValueError(
                "Binary gradient requires observable sensitivities. For "
                "adjoint sensitivity analysis, reconstruct sy before calling "
                "calculate_gradients."
            )

        y = self._extract_y(problem, sim)
        z = problem.labels
        X = self._design_matrix(problem, y)
        beta = x_inner[problem.n_alpha :]
        beta_for_meas = beta[problem.beta_group_ixs]
        p = expit(X @ x_inner)

        # Per-measurement gradient weight. Unpenalised term: (p-z)*beta.
        scale = (p - z) * beta_for_meas
        if self.use_firth:
            # Firth penalty depends on θ via the y-covariate, adding -½∂log|I|/∂θ
            # = -Σ_i c_i sy_i. General over beta-groups: g_i picks measurement i's
            # own beta-column of I^{-1}. (All vectorised; no parameter-dim loops.)
            w = p * (1.0 - p)
            _, B, q, _ = self._firth_info(X, w)  # B = I^{-1} Xᵀ  (P×n)
            x_iinv = B.T  # (n, P) = X I^{-1}
            beta_cols = problem.n_alpha + np.arange(problem.n_beta)
            g = x_iinv[:, beta_cols][np.arange(problem.n_meas), problem.beta_group_ixs]
            c = w * (g + 0.5 * beta_for_meas * (1.0 - 2.0 * p) * q)
            scale = scale - c

        sim_grads = [np.zeros(len(par_sim_ids)) for _ in sim]
        plists = [
            np.asarray(plist, dtype=int) for plist in par_edatas_indices
        ]

        for i in range(problem.n_meas):
            ci = problem.cond_ixs[i]
            ti = problem.time_ixs[i]
            oi = problem.obs_ixs[i]
            sim_grads[ci][plists[ci]] += scale[i] * sy[ci][ti, :, oi]

        for ci, sim_grad in enumerate(sim_grads):
            add_sim_grad_to_opt_grad(
                par_opt_ids=par_opt_ids,
                par_sim_ids=par_sim_ids,
                condition_map_sim_var=parameter_mapping[ci].map_sim_var,
                sim_grad=sim_grad,
                opt_grad=snllh,
            )

        return snllh

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _design_matrix(
        problem: BinaryInnerProblem,
        y: np.ndarray,
    ) -> np.ndarray:
        """Logistic design matrix X, shape ``(n_meas, n_alpha + n_beta)``.

        Alpha-group indicator columns (theta-independent) followed by
        beta-group columns holding ``y_i`` (theta-dependent). Then
        ``eta = X @ x_inner == alpha_{g(i)} + beta_{b(i)} * y_i``.
        """
        n = problem.n_meas
        X = np.zeros((n, problem.n_alpha + problem.n_beta))
        rows = np.arange(n)
        X[rows, problem.alpha_group_ixs] = 1.0
        X[rows, problem.n_alpha + problem.beta_group_ixs] = y
        return X

    @staticmethod
    def _firth_info(
        X: np.ndarray,
        w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Fisher information and the pieces the Firth terms need.

        Returns ``(I, B, q, logdet)`` where ``I = XᵀWX``, ``B = I⁻¹Xᵀ`` (P×n,
        via a solve for stability), ``q_i = x_iᵀ I⁻¹ x_i`` are the unweighted
        leverages, and ``logdet = log|I|`` (via ``slogdet``). A tiny diagonal
        jitter guards against mid-iteration singularity (W → 0 as p saturates).
        """
        n_inner = X.shape[1]
        info = (X * w[:, None]).T @ X
        info = info + 1e-10 * np.eye(n_inner)
        _, logdet = np.linalg.slogdet(info)
        B = np.linalg.solve(info, X.T)  # I⁻¹ Xᵀ, (P, n)
        q = np.einsum("ij,ji->i", X, B)  # diag(X I⁻¹ Xᵀ), (n,)
        return info, B, q, float(logdet)

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
