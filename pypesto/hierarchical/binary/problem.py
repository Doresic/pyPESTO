"""Inner problem for binary hierarchical optimization."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..base_problem import AmiciInnerProblem
from .parameter import BinaryInnerParameter, BinaryInnerParameterType

try:
    import amici
    import petab.v1 as petab
    from petab.v1.C import (
        OBSERVABLE_ID,
        OBSERVABLE_PARAMETERS,
        SIMULATION_CONDITION_ID,
        TIME,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)

#: Value of the ``measurementType`` PEtab column for binary measurements.
BINARY_MEASUREMENT_TYPE = "BINARY"
#: PEtab column name for measurement type.
MEASUREMENT_TYPE_COL = "measurementType"
#: PEtab column assigning binary measurements to intercept groups.
BINARY_ALPHA_GROUP_COL = "binaryAlphaGroupId"
#: Optional PEtab column assigning binary measurements to slope groups.
BINARY_BETA_GROUP_COL = "binaryBetaGroupId"
#: Default beta group if no beta grouping column is present.
DEFAULT_BETA_GROUP_ID = "global"


class BinaryInnerProblem(AmiciInnerProblem):
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
    beta_ids:
        Ordered list of beta parameter IDs (β_0, β_1, …, β_{B-1}).
    n_alpha:
        Number of distinct alpha groups G.
    n_beta:
        Number of distinct beta groups B.
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
    beta_group_ixs:
        Shape ``(n_meas,)`` int — which beta parameter (0-based index
        into ``beta_ids``) applies to each binary measurement.
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
        beta_group_ixs: np.ndarray,
        alpha_ids: list[str],
        beta_ids: list[str],
        edatas: list,
        data: list[np.ndarray],
    ):
        super().__init__(xs=xs, data=data, edatas=edatas)
        self.alpha_ids: list[str] = list(alpha_ids)
        self.beta_ids: list[str] = list(beta_ids)
        self.n_alpha: int = len(alpha_ids)
        self.n_beta: int = len(beta_ids)
        self.n_meas: int = int(len(labels))

        self.cond_ixs: np.ndarray = np.asarray(cond_ixs, dtype=int)
        self.time_ixs: np.ndarray = np.asarray(time_ixs, dtype=int)
        self.obs_ixs: np.ndarray = np.asarray(obs_ixs, dtype=int)
        self.labels: np.ndarray = np.asarray(labels, dtype=float)
        self.alpha_group_ixs: np.ndarray = np.asarray(alpha_group_ixs, dtype=int)
        self.beta_group_ixs: np.ndarray = np.asarray(beta_group_ixs, dtype=int)
        # NOTE: do NOT store `self.edatas = edatas` here. AMICI ExpData are SWIG objects, so
        # keeping a reference makes the binary objective unpicklable and breaks MultiProcessEngine.
        # The base AmiciInnerProblem (super().__init__ above) already extracts the data it needs
        # and discards the edatas, matching the relative/ordinal inner problems.

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_interpretable_x_ids(self) -> list[str]:
        """Return interpretable inner parameter IDs."""
        return []

    def get_interpretable_x_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return bounds for interpretable inner parameters."""
        return np.asarray([]), np.asarray([])

    def get_interpretable_x_scales(self) -> list[str]:
        """Return scales for interpretable inner parameters."""
        return []

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lb, ub) arrays for inner parameters.

        Order: ``[α_0, …, α_{G-1}, β_0, …, β_{B-1}]``.
        """
        ids = self.alpha_ids + self.beta_ids
        lb = np.array([self.xs[i].lb for i in ids])
        ub = np.array([self.xs[i].ub for i in ids])
        return lb, ub

    def initialize(self) -> None:
        """Initialize the subproblem."""
        for x in self.xs.values():
            x.initialize()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_petab_amici(
        cls,
        petab_problem: "petab.Problem",
        amici_model: "amici.Model",
        edatas: list,
        beta_lb: float = -np.inf,
        beta_ub: float = np.inf,
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
        beta_lb:
            Lower bound applied to every BETA inner parameter (the shared
            binary slope β). Default ``-inf`` (unconstrained). Box-constraining
            β floors the outer θ-gradient ``β·Σ(p_i−z_i)·∂y_i/∂θ`` so the KO
            data keeps informing θ instead of self-extinguishing as β→0.
        beta_ub:
            Upper bound applied to every BETA inner parameter. Default
            ``+inf``. β<0 by convention, so ``β ∈ [−hi, −lo]`` enforces
            ``|β| ∈ [lo, hi]``.
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

        cls._validate_binary_measurements(binary_df)

        # --- parse rows ---
        alpha_group_to_ix: dict[str, int] = {}
        beta_group_to_ix: dict[str, int] = {}
        # Replicate-aware timepoint mapping. The edata stores replicates as REPEATED timepoint
        # rows; hand out a DISTINCT row index per (condition, time value) so every replicate cell
        # is marked in `ixs`. The old code mapped every steady-state (inf) replicate to the LAST
        # row (len(ts)-1), so replicates collapsed onto one cell and the unmarked cells leaked into
        # the quantitative (Gaussian) likelihood -> binary measurements double-counted -> biased.
        tp_groups: dict[int, dict] = {}
        tp_cursor: dict[int, dict] = {}

        rows: list[tuple[int, int, int, float, int, int, str, str]] = []
        # each entry:
        # (cond_ix, time_ix, obs_ix, label, alpha_group_ix, beta_group_ix,
        #  alpha_group_id, beta_group_id)

        for _, row in binary_df.iterrows():
            cond_id = str(row[SIMULATION_CONDITION_ID])
            obs_id = str(row[OBSERVABLE_ID])
            time = float(row[TIME])
            label = float(row["measurement"])

            alpha_group_id = str(row[BINARY_ALPHA_GROUP_COL]).strip()
            beta_group_id = (
                str(row[BINARY_BETA_GROUP_COL]).strip()
                if BINARY_BETA_GROUP_COL in binary_df.columns
                and not pd.isna(row[BINARY_BETA_GROUP_COL])
                and str(row[BINARY_BETA_GROUP_COL]).strip()
                else DEFAULT_BETA_GROUP_ID
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
            # map each measurement (INCLUDING replicates) to its own distinct timepoint row
            tkey = "inf" if np.isinf(time) else round(float(time), 12)
            if cond_ix not in tp_groups:
                grp: dict = {}
                for _idx, _tv in enumerate(ts):
                    _k = "inf" if np.isinf(_tv) else round(float(_tv), 12)
                    grp.setdefault(_k, []).append(_idx)
                tp_groups[cond_ix] = grp
            avail = tp_groups[cond_ix].get(tkey, [])
            if not avail:
                raise ValueError(
                    f"Timepoint {time} for condition '{cond_id}' not "
                    f"found in edata timepoints {ts}."
                )
            cur = tp_cursor.setdefault(cond_ix, {}).get(tkey, 0)
            # consecutive distinct rows for replicates; clamp if more meas than rows
            time_ix = avail[cur] if cur < len(avail) else avail[-1]
            tp_cursor[cond_ix][tkey] = cur + 1

            if alpha_group_id not in alpha_group_to_ix:
                alpha_group_to_ix[alpha_group_id] = len(alpha_group_to_ix)
            alpha_group_ix = alpha_group_to_ix[alpha_group_id]

            if beta_group_id not in beta_group_to_ix:
                beta_group_to_ix[beta_group_id] = len(beta_group_to_ix)
            beta_group_ix = beta_group_to_ix[beta_group_id]

            rows.append(
                (
                    cond_ix,
                    time_ix,
                    obs_ix,
                    label,
                    alpha_group_ix,
                    beta_group_ix,
                    alpha_group_id,
                    beta_group_id,
                )
            )

        # --- build flat arrays ---
        n_meas = len(rows)
        n_conds = len(edatas)
        n_obs = len(obs_ids)

        cond_ixs = np.array([r[0] for r in rows], dtype=int)
        time_ixs = np.array([r[1] for r in rows], dtype=int)
        obs_ixs = np.array([r[2] for r in rows], dtype=int)
        labels = np.array([r[3] for r in rows], dtype=float)
        alpha_group_ixs = np.array([r[4] for r in rows], dtype=int)
        beta_group_ixs = np.array([r[5] for r in rows], dtype=int)

        # ordered inner parameter IDs (stable: insertion order of group maps)
        alpha_group_ids: list[str] = sorted(
            alpha_group_to_ix.keys(), key=lambda k: alpha_group_to_ix[k]
        )
        beta_group_ids: list[str] = sorted(
            beta_group_to_ix.keys(), key=lambda k: beta_group_to_ix[k]
        )
        alpha_ids = [f"alpha__{gid}" for gid in alpha_group_ids]
        beta_ids = [f"beta__{gid}" for gid in beta_group_ids]

        # --- build ixs (per-condition bool arrays) ---
        n_timepoints_per_cond = [len(ts) for ts in edata_timepoints]
        data = [
            amici.numpy.ExpDataView(edata)["observedData"] for edata in edatas
        ]

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
        for gid, aid in zip(alpha_group_ids, alpha_ids, strict=True):
            mask = np.array([r[6] == gid for r in rows])
            xs.append(
                BinaryInnerParameter(
                    inner_parameter_id=aid,
                    inner_parameter_type=BinaryInnerParameterType.ALPHA,
                    ixs=_build_ixs(mask),
                )
            )

        for gid, bid in zip(beta_group_ids, beta_ids, strict=True):
            mask = np.array([r[7] == gid for r in rows])
            xs.append(
                BinaryInnerParameter(
                    inner_parameter_id=bid,
                    inner_parameter_type=BinaryInnerParameterType.BETA,
                    lb=beta_lb,
                    ub=beta_ub,
                    ixs=_build_ixs(mask),
                )
            )

        return cls(
            xs=xs,
            cond_ixs=cond_ixs,
            time_ixs=time_ixs,
            obs_ixs=obs_ixs,
            labels=labels,
            alpha_group_ixs=alpha_group_ixs,
            beta_group_ixs=beta_group_ixs,
            alpha_ids=alpha_ids,
            beta_ids=beta_ids,
            edatas=edatas,
            data=data,
        )

    @staticmethod
    def _validate_binary_measurements(binary_df: "pd.DataFrame") -> None:
        """Validate binary measurement-table conventions."""
        labels = binary_df["measurement"].astype(float)
        invalid_labels = ~labels.isin([0.0, 1.0])
        if invalid_labels.any():
            bad = binary_df.loc[
                invalid_labels,
                [SIMULATION_CONDITION_ID, OBSERVABLE_ID, "measurement"],
            ].head()
            raise ValueError(
                "Binary measurements must have measurement values 0 or 1. "
                f"Invalid examples:\n{bad}"
            )

        if BINARY_ALPHA_GROUP_COL not in binary_df.columns:
            raise ValueError(
                f"Binary measurements require column "
                f"'{BINARY_ALPHA_GROUP_COL}' to define alpha sharing."
            )
        alpha_groups = binary_df[BINARY_ALPHA_GROUP_COL]
        if alpha_groups.isna().any() or (
            alpha_groups.astype(str).str.strip() == ""
        ).any():
            raise ValueError(
                f"Binary measurements require non-empty "
                f"'{BINARY_ALPHA_GROUP_COL}' values."
            )

        if OBSERVABLE_PARAMETERS in binary_df.columns:
            obs_pars = binary_df[OBSERVABLE_PARAMETERS]
            has_obs_pars = obs_pars.notna() & (
                obs_pars.astype(str).str.strip() != ""
            )
            if has_obs_pars.any():
                bad = binary_df.loc[
                    has_obs_pars,
                    [SIMULATION_CONDITION_ID, OBSERVABLE_ID, OBSERVABLE_PARAMETERS],
                ].head()
                raise ValueError(
                    "Binary measurements must not use observableParameters; "
                    "alpha/beta are binary inner parameters. Invalid examples:"
                    f"\n{bad}"
                )
