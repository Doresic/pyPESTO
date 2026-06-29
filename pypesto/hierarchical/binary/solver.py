"""Inner solver for binary hierarchical optimization."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr, ndtr  # σ(η); log Φ(η); Φ(η)

from ..base_solver import InnerSolver
from ...objective.amici.amici_util import add_sim_grad_to_opt_grad
from .problem import BinaryInnerProblem

try:
    from amici.petab.parameter_mapping import ParameterMapping
except ImportError:
    pass

logger = logging.getLogger(__name__)

LARGE_INNER_PARAMETER_WARNING_THRESHOLD = 50.0

#: ½·log(2π); used by the probit log-pdf (log φ = −½η² − this).
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


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

    def __init__(self, use_firth: bool = True, binary_link: str = "logit"):
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
        binary_link:
            Link function for the Bernoulli mean ``p = link⁻¹(η)``.
            ``"logit"`` (default) uses ``p = σ(η)``; ``"probit"`` uses
            ``p = Φ(η)``. The choice is link-agnostic everywhere except the
            ``_p`` / ``_score`` / ``_weight`` helpers and the Firth correction.
        """
        super().__init__()
        self.use_firth = use_firth
        if binary_link not in ("logit", "probit"):
            raise ValueError(
                f"binary_link must be 'logit' or 'probit', got {binary_link!r}."
            )
        #: Link function for the Bernoulli mean ``p = link⁻¹(η)``. ``"logit"``
        #: (σ, the default and original behaviour) or ``"probit"`` (Φ). All
        #: link-specific math lives in the ``_p`` / ``_score`` / ``_weight``
        #: dispatch helpers; everything else is link-agnostic.
        self.binary_link = binary_link

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
        link = self.binary_link

        def _nll_and_grad(x_inner: np.ndarray) -> tuple[float, np.ndarray]:
            eta = X @ x_inner
            p = self._p(eta, link)
            p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
            nll = -np.sum(z * np.log(p_clip) + (1.0 - z) * np.log(1.0 - p_clip))
            resid_eff = self._score(eta, z, link)  # gradient weight per meas.
            if self.use_firth:
                # Firth: J = NLL - ½log|I|, I = XᵀWX. The penalty adds the
                # general modified-score term  −½ w'(η) q  to the per-point
                # score (q_i = leverage). For logit this is the byte-identical
                # closed form −w q (½−p); probit uses the general ½ w' q.
                w = self._weight(eta, link)
                _, _, q, logdet = self._firth_info(X, w)
                nll = nll - 0.5 * logdet
                if link == "logit":
                    resid_eff = resid_eff - w * q * (0.5 - p)
                else:
                    resid_eff = resid_eff - 0.5 * self._weight_deriv(eta, link) * q
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
        eta = X @ x_inner
        p = self._p(eta, self.binary_link)
        p_clip = np.clip(p, 1e-15, 1.0 - 1e-15)
        nllh = -np.sum(z * np.log(p_clip) + (1.0 - z) * np.log(1.0 - p_clip))
        if self.use_firth:
            # Outer FVAL must be the penalised objective so the envelope theorem
            # gives dF/dθ = ∂(NLL - ½log|I|)/∂θ cleanly.
            w = self._weight(eta, self.binary_link)
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
        eta = X @ x_inner
        link = self.binary_link
        p = self._p(eta, link)

        # Per-measurement gradient weight. Unpenalised term: score(η)*beta.
        scale = self._score(eta, z, link) * beta_for_meas
        if self.use_firth:
            # Firth penalty depends on θ via the y-covariate, adding -½∂log|I|/∂θ
            # = -Σ_i c_i sy_i. General over beta-groups: g_i picks measurement i's
            # own beta-column of I^{-1}. (All vectorised; no parameter-dim loops.)
            # General c_i = w_i g_i + β_i·(½ w'_i q_i); logit keeps its byte-
            # identical closed form, probit uses the general ½ w' q.
            w = self._weight(eta, link)
            _, B, q, _ = self._firth_info(X, w)  # B = I^{-1} Xᵀ  (P×n)
            x_iinv = B.T  # (n, P) = X I^{-1}
            beta_cols = problem.n_alpha + np.arange(problem.n_beta)
            g = x_iinv[:, beta_cols][np.arange(problem.n_meas), problem.beta_group_ixs]
            if link == "logit":
                c = w * (g + 0.5 * beta_for_meas * (1.0 - 2.0 * p) * q)
            else:
                c = w * g + beta_for_meas * (0.5 * self._weight_deriv(eta, link) * q)
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
    # Link dispatch (the ONLY link-specific math)
    # ------------------------------------------------------------------
    #
    # All links share the same Bernoulli NLL and the same Firth penalty
    # skeleton (½log|I|, I = XᵀWX). A link is fully specified by:
    #     p(η)   the mean (probability of z=1),
    #     s(η)   the per-point NLL score  ∂(−ℓ)/∂η,
    #     w(η)   the Fisher working weight (the W in I = XᵀWX),
    # plus, for the Firth correction, w'(η) = dw/dη.
    #
    #   logit:  p = σ(η);  s = p − z;   w = p(1−p);     w' = w(1−2p)
    #   probit: p = Φ(η);  s = (1−z)·r̄ − z·r;
    #                      w = φ²/(Φ(1−Φ)) = r·r̄;   w' = w(r̄ − r − 2η)
    #   with r = φ/Φ and r̄ = φ/(1−Φ) (computed in the log domain for tail
    #   stability; each grows only ~|η|, so no overflow).
    # See _scratch/probit_link/PROBIT_DERIVATION.md for the full derivation.

    @staticmethod
    def _p(eta: np.ndarray, link: str = "logit") -> np.ndarray:
        """Bernoulli mean ``p = link⁻¹(η)``.

        ``"logit"`` → ``σ(η) = expit(η)`` (default, original behaviour);
        ``"probit"`` → ``Φ(η)``.
        """
        if link == "logit":
            return expit(eta)
        if link == "probit":
            return ndtr(eta)
        raise ValueError(f"Unknown binary_link '{link}'.")

    @staticmethod
    def _score(
        eta: np.ndarray, z: np.ndarray, link: str = "logit"
    ) -> np.ndarray:
        """Per-point NLL score ``s_i = ∂(−ℓ_i)/∂η_i``.

        ``"logit"`` (canonical) → ``s = p − z``.
        ``"probit"`` → ``s = (1−z)·φ/(1−Φ) − z·φ/Φ``.
        """
        if link == "logit":
            return expit(eta) - z
        if link == "probit":
            r, rbar = BinaryInnerSolver._probit_mills(eta)
            return (1.0 - z) * rbar - z * r
        raise ValueError(f"Unknown binary_link '{link}'.")

    @staticmethod
    def _weight(eta: np.ndarray, link: str = "logit") -> np.ndarray:
        """Fisher working weight ``w_i`` (the ``W`` in ``I = XᵀWX``).

        ``"logit"`` → ``w = p(1 − p)``; ``"probit"`` → ``w = φ²/(Φ(1−Φ))``.
        """
        if link == "logit":
            p = expit(eta)
            return p * (1.0 - p)
        if link == "probit":
            r, rbar = BinaryInnerSolver._probit_mills(eta)
            return r * rbar
        raise ValueError(f"Unknown binary_link '{link}'.")

    @staticmethod
    def _weight_deriv(eta: np.ndarray, link: str = "logit") -> np.ndarray:
        """Derivative of the working weight, ``w'(η) = dw/dη``.

        Enters only the Firth modified-score correction (the general term is
        ``½ w' q``). The logit production path keeps its byte-identical closed
        form ``−w q (½−p)`` and does NOT call this; both branches are provided
        so the helper is complete and directly finite-difference testable.

        ``"logit"`` → ``w' = w(1 − 2p)``.
        ``"probit"`` → ``w' = w(r̄ − r − 2η)`` with ``r=φ/Φ``, ``r̄=φ/(1−Φ)``.
        """
        if link == "logit":
            p = expit(eta)
            w = p * (1.0 - p)
            return w * (1.0 - 2.0 * p)
        if link == "probit":
            r, rbar = BinaryInnerSolver._probit_mills(eta)
            w = r * rbar
            return w * (rbar - r - 2.0 * eta)
        raise ValueError(f"Unknown binary_link '{link}'.")

    @staticmethod
    def _probit_mills(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(r, r̄) = (φ/Φ, φ/(1−Φ))`` for the probit link.

        Evaluated in the log domain via ``log_ndtr`` (``log Φ``) so both stay
        finite and accurate deep into either tail. Note ``1−Φ(η) = Φ(−η)``,
        so ``r̄ = φ/Φ(−η)``.
        """
        log_phi = -0.5 * eta * eta - _LOG_SQRT_2PI
        r = np.exp(log_phi - log_ndtr(eta))       # φ/Φ
        rbar = np.exp(log_phi - log_ndtr(-eta))   # φ/(1−Φ)
        return r, rbar

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
