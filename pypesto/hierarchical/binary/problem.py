"""Inner problem for binary hierarchical optimization."""

from __future__ import annotations

import logging

import numpy as np

from .parameter import BinaryInnerParameter, BinaryInnerParameterType

try:
    import amici
    import petab.v1 as petab
    from petab.v1.C import (
        LOWER_BOUND,
        OBSERVABLE_ID,
        OBSERVABLE_PARAMETERS,
        PARAMETER_SEPARATOR,
        SIMULATION_CONDITION_ID,
        TIME,
        UPPER_BOUND,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)

#: Value of the ``measurementType`` PEtab column for binary measurements.
BINARY_MEASUREMENT_TYPE = "BINARY"
#: PEtab column name for measurement type.
MEASUREMENT_TYPE_COL = "measurementType"


class BinaryInnerProblem:
    """Inner optimization problem for binary (Bernoulli) data.

    Stores all static information needed by :class:`BinaryInnerSolver`:
    the inner parameters, flat index arrays that locate each binary
    measurement within the AMICI rdata arrays, and the binary labels.

    Attributes
    ----------
    xs:
        Mapping ``parameter_id → BinaryInnerParameter`` for all inner
        parameters (all α groups and the single β).
    alpha_ids:
        Ordered list of alpha parameter IDs (α_0, α_1, …, α_{G-1}).
    beta_id:
        The shared slope parameter ID.
    n_alpha:
        Number of distinct alpha groups G.
    n_meas:
        Total number of binary measurements.
    cond_ixs:
        Shape ``(n_meas,)`` int — index into the edatas / rdatas list for
        each binary measurement.
    time_ixs:
        Shape ``(n_meas,)`` int — timepoint row index in ``rdata.y`` for
        each binary measurement.
    obs_ixs:
        Shape ``(n_meas,)`` int — observable column index in ``rdata.y``
        for each binary measurement.
    labels:
        Shape ``(n_meas,)`` float — observed binary label (0.0 or 1.0).
    alpha_group_ixs:
        Shape ``(n_meas,)`` int — which alpha parameter (0-based index
        into ``alpha_ids``) applies to each binary measurement.
    edatas:
        AMICI ``ExpData`` objects (same list as passed to AMICI).
    """

    def __init__(
        self,
        xs: list[BinaryInnerParameter],
        cond_ixs: np.ndarray,
        time_ixs: np.ndarray,
        obs_ixs: np.ndarray,
        labels: np.ndarray,
        alpha_group_ixs: np.ndarray,
        alpha_ids: list[str],
        beta_id: str,
        edatas: list,
    ):
        self.xs: dict[str, BinaryInnerParameter] = {
            x.inner_parameter_id: x for x in xs
        }
        self.alpha_ids: list[str] = list(alpha_ids)
        self.beta_id: str = beta_id
        self.n_alpha: int = len(alpha_ids)
        self.n_meas: int = int(len(labels))

        self.cond_ixs: np.ndarray = np.asarray(cond_ixs, dtype=int)
        self.time_ixs: np.ndarray = np.asarray(time_ixs, dtype=int)
        self.obs_ixs: np.ndarray = np.asarray(obs_ixs, dtype=int)
        self.labels: np.ndarray = np.asarray(labels, dtype=float)
        self.alpha_group_ixs: np.ndarray = np.asarray(alpha_group_ixs, dtype=int)

        self.edatas = edatas

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_x_ids(self) -> list[str]:
        """Return all inner parameter IDs."""
        return list(self.xs.keys())

    def is_empty(self) -> bool:
        """Return ``True`` if no binary measurements exist."""
        return self.n_meas == 0

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lb, ub) arrays for inner parameters.

        Order: ``[α_0, …, α_{G-1}, β]``.
        """
        ids = self.alpha_ids + [self.beta_id]
        lb = np.array([self.xs[i].lb for i in ids])
        ub = np.array([self.xs[i].ub for i in ids])
        return lb, ub

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_petab_amici(
        cls,
        petab_problem: "petab.Problem",
        amici_model: "amici.Model",
        edatas: list,
    ) -> "BinaryInnerProblem":
        """Construct from a PEtab problem and AMICI objects.

        Reads the measurements table, identifies rows with
        ``measurementType == "BINARY"``, and builds the flat index arrays
        and :class:`BinaryInnerParameter` objects.

        Parameters
        ----------
        petab_problem:
            The PEtab problem (must have a ``measurement_df`` and
            ``parameter_df``).
        amici_model:
            The compiled AMICI model (used for observable ID → index
            mapping).
        edatas:
            The AMICI ``ExpData`` list in the same order as the
            simulations will be run.  ``edata.id`` must match PEtab
            ``simulationConditionId``.
        """
        meas_df = petab_problem.measurement_df

        # --- filter binary rows ---
        if MEASUREMENT_TYPE_COL not in meas_df.columns:
            raise ValueError(
                f"Column '{MEASUREMENT_TYPE_COL}' not found in measurements "
                "table. Binary measurements require this column set to "
                f"'{BINARY_MEASUREMENT_TYPE}'."
            )
        binary_df = meas_df[
            meas_df[MEASUREMENT_TYPE_COL] == BINARY_MEASUREMENT_TYPE
        ]
        if binary_df.empty:
            raise ValueError(
                f"No measurements with measurementType == "
                f"'{BINARY_MEASUREMENT_TYPE}' found."
            )

        # --- build look-up tables ---
        cond_id_to_ix: dict[str, int] = {
            edata.id: i for i, edata in enumerate(edatas)
        }
        obs_ids: list[str] = list(amici_model.getObservableIds())
        obs_id_to_ix: dict[str, int] = {
            oid: i for i, oid in enumerate(obs_ids)
        }
        # timepoints per condition (list of floats)
        edata_timepoints: list[list[float]] = [
            list(edata.getTimepoints()) for edata in edatas
        ]

        # --- parse rows ---
        alpha_id_to_group: dict[str, int] = {}
        beta_id: str | None = None

        rows: list[tuple[int, int, int, float, int, str]] = []
        # each entry: (cond_ix, time_ix, obs_ix, label, alpha_group_ix, alpha_id)

        for _, row in binary_df.iterrows():
            cond_id = str(row[SIMULATION_CONDITION_ID])
            obs_id = str(row[OBSERVABLE_ID])
            time = float(row[TIME])
            label = float(row["measurement"])

            obs_params_raw = str(row[OBSERVABLE_PARAMETERS])
            parts = [p.strip() for p in obs_params_raw.split(PARAMETER_SEPARATOR)]
            if len(parts) < 2:
                raise ValueError(
                    f"Binary measurement for condition '{cond_id}' has "
                    f"observableParameters='{obs_params_raw}'. Expected two "
                    "semicolon-separated IDs: 'alpha_id;beta_id'."
                )
            alpha_id, b_id = parts[0], parts[1]

            if beta_id is None:
                beta_id = b_id
            elif beta_id != b_id:
                raise ValueError(
                    f"Multiple beta parameter IDs found in binary "
                    f"measurements: '{beta_id}' and '{b_id}'. Only one beta "
                    "parameter is supported."
                )

            if cond_id not in cond_id_to_ix:
                raise KeyError(
                    f"simulationConditionId '{cond_id}' from binary "
                    "measurements not found in edatas."
                )
            cond_ix = cond_id_to_ix[cond_id]

            if obs_id not in obs_id_to_ix:
                raise KeyError(
                    f"observableId '{obs_id}' not found in AMICI model "
                    f"observables: {obs_ids}"
                )
            obs_ix = obs_id_to_ix[obs_id]

            ts = edata_timepoints[cond_ix]
            if np.isinf(time):
                # steady-state: last entry in timepoints list
                time_ix = len(ts) - 1
            else:
                try:
                    time_ix = ts.index(time)
                except ValueError:
                    raise ValueError(
                        f"Timepoint {time} for condition '{cond_id}' not "
                        f"found in edata timepoints {ts}."
                    ) from None

            if alpha_id not in alpha_id_to_group:
                alpha_id_to_group[alpha_id] = len(alpha_id_to_group)
            alpha_group_ix = alpha_id_to_group[alpha_id]

            rows.append((cond_ix, time_ix, obs_ix, label, alpha_group_ix, alpha_id))

        if beta_id is None:
            raise ValueError("No beta parameter found in binary measurements.")

        # --- build flat arrays ---
        n_meas = len(rows)
        n_conds = len(edatas)
        n_obs = len(obs_ids)

        cond_ixs = np.array([r[0] for r in rows], dtype=int)
        time_ixs = np.array([r[1] for r in rows], dtype=int)
        obs_ixs = np.array([r[2] for r in rows], dtype=int)
        labels = np.array([r[3] for r in rows], dtype=float)
        alpha_group_ixs = np.array([r[4] for r in rows], dtype=int)

        # ordered alpha IDs (stable: insertion order of alpha_id_to_group)
        alpha_ids: list[str] = sorted(
            alpha_id_to_group.keys(), key=lambda k: alpha_id_to_group[k]
        )

        # --- bounds from parameters.tsv ---
        params_df = petab_problem.parameter_df

        def _get_bounds(pid: str) -> tuple[float, float]:
            if pid in params_df.index:
                r = params_df.loc[pid]
                lb = float(r.get(LOWER_BOUND, -np.inf))
                ub = float(r.get(UPPER_BOUND, np.inf))
            else:
                lb, ub = -np.inf, np.inf
            return lb, ub

        # --- build ixs (per-condition bool arrays) ---
        n_timepoints_per_cond = [len(ts) for ts in edata_timepoints]

        def _build_ixs(meas_mask: np.ndarray) -> list[np.ndarray]:
            ixs = [
                np.zeros((n_timepoints_per_cond[c], n_obs), dtype=bool)
                for c in range(n_conds)
            ]
            for i in np.where(meas_mask)[0]:
                ixs[cond_ixs[i]][time_ixs[i], obs_ixs[i]] = True
            return ixs

        # --- build BinaryInnerParameter objects ---
        xs: list[BinaryInnerParameter] = []
        for aid in alpha_ids:
            lb, ub = _get_bounds(aid)
            mask = np.array([r[5] == aid for r in rows])
            xs.append(
                BinaryInnerParameter(
                    inner_parameter_id=aid,
                    inner_parameter_type=BinaryInnerParameterType.ALPHA,
                    lb=lb,
                    ub=ub,
                    ixs=_build_ixs(mask),
                )
            )

        lb_beta, ub_beta = _get_bounds(beta_id)
        xs.append(
            BinaryInnerParameter(
                inner_parameter_id=beta_id,
                inner_parameter_type=BinaryInnerParameterType.BETA,
                lb=lb_beta,
                ub=ub_beta,
                ixs=_build_ixs(np.ones(n_meas, dtype=bool)),
            )
        )

        return cls(
            xs=xs,
            cond_ixs=cond_ixs,
            time_ixs=time_ixs,
            obs_ixs=obs_ixs,
            labels=labels,
            alpha_group_ixs=alpha_group_ixs,
            alpha_ids=alpha_ids,
            beta_id=beta_id,
            edatas=edatas,
        )
