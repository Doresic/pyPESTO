"""Binary hierarchical inner optimization submodule.

Implements a Bernoulli-likelihood inner problem for integrating binary
qualitative data (e.g. CRISPR gene-essentiality screens) into ODE model
parameter estimation via hierarchical optimization.

Mathematical formulation
------------------------
For each binary measurement i, the model outputs an observable ``y_i(θ)``.
Two inner parameter families are estimated for groups of measurements:

- ``α_g`` — baseline death propensity for group g (e.g. cell line)
- ``β_b`` — logistic slope for beta group b (global by default)

Death probability:   ``p_i = σ(α_{g_i} + β_{b_i} · y_i)``
Likelihood:          ``ℓ = Σ_i [z_i log p_i + (1−z_i) log(1−p_i)]``

The inner optimization over (α, β) is a logistic regression, solved
numerically via scipy L-BFGS-B.  The outer gradient is obtained via the
envelope theorem:

    ∂(−ℓ)/∂θ_k = Σ_i (p*_i − z_i) · β*_{b_i} · sy_i_k

where ``sy_i_k = ∂y_i/∂θ_k`` comes from AMICI forward sensitivities or
from the Frohlich single-datapoint adjoint reconstruction.

PEtab representation
--------------------
Binary measurements are identified by ``measurementType = "BINARY"`` in
the measurements table.  Binary rows must have ``measurement`` values in
``{0, 1}``, empty ``observableParameters``, and a ``binaryAlphaGroupId``.
The optional ``binaryBetaGroupId`` controls slope sharing; if absent or empty,
one global beta is used.  α/β are internal inner parameters, not PEtab
observable parameters and not outer parameters.
"""

from .calculator import BinaryAmiciCalculator
from .parameter import BinaryInnerParameter, BinaryInnerParameterType
from .problem import (
    BINARY_ALPHA_GROUP_COL,
    BINARY_BETA_GROUP_COL,
    BinaryInnerProblem,
)
from .solver import BinaryInnerSolver

#: Value of the ``measurementType`` column that flags binary measurements.
BINARY_MEASUREMENT_TYPE = "BINARY"

__all__ = [
    "BinaryAmiciCalculator",
    "BinaryInnerParameter",
    "BinaryInnerParameterType",
    "BinaryInnerProblem",
    "BinaryInnerSolver",
    "BINARY_MEASUREMENT_TYPE",
    "BINARY_ALPHA_GROUP_COL",
    "BINARY_BETA_GROUP_COL",
]
