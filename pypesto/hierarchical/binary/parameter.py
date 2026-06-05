"""Inner parameter class for binary hierarchical optimization."""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class BinaryInnerParameterType(str, Enum):
    """Type of a binary inner parameter."""

    ALPHA = "alpha"  # group-specific baseline death propensity (α_g)
    BETA = "beta"    # shared logistic slope (β)


class BinaryInnerParameter:
    """Inner parameter for the binary hierarchical likelihood.

    Duck-typed to the same interface as ``InnerParameter`` but independent
    of that base class to avoid coupling to ``InnerParameterType`` in
    ``pypesto.C``.

    Attributes
    ----------
    inner_parameter_id:
        Unique string identifier (matches the PEtab parameter ID).
    inner_parameter_type:
        ``ALPHA`` (group baseline) or ``BETA`` (shared slope).
    scale:
        Parameter scale for optimization.  Currently only ``'lin'``
        (linear scale) is supported.
    lb:
        Lower bound for the inner optimization.
    ub:
        Upper bound for the inner optimization.
    ixs:
        One boolean array per simulation condition, each of shape
        ``(n_timepoints, n_observables)``.  Entry ``[t, o]`` is ``True``
        when the observable at that (condition, timepoint, observable)
        index contributes to this parameter's likelihood term.
        Convention matches all other hierarchical submodules.
    dummy_value:
        Placeholder value used before the inner optimization has run.
    value:
        Current optimal value (updated in-place by the solver).
    """

    def __init__(
        self,
        inner_parameter_id: str,
        inner_parameter_type: BinaryInnerParameterType,
        scale: str = "lin",
        lb: float = -np.inf,
        ub: float = np.inf,
        ixs: list[np.ndarray] | None = None,
        dummy_value: float = 0.0,
    ):
        self.inner_parameter_id: str = inner_parameter_id
        self.inner_parameter_type: BinaryInnerParameterType = inner_parameter_type
        self.scale: str = scale
        self.lb: float = lb
        self.ub: float = ub
        self.ixs: list[np.ndarray] | None = ixs
        self.dummy_value: float = dummy_value
        self.value: float = dummy_value

    def initialize(self) -> None:
        """Reset the parameter to its dummy value."""
        self.value = self.dummy_value
