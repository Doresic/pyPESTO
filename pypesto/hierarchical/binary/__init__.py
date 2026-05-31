"""Binary hierarchical inner optimization submodule.

Implements a Bernoulli-likelihood inner problem for integrating binary
qualitative data (e.g. CRISPR gene-essentiality screens) into ODE model
parameter estimation via hierarchical optimization.

Mathematical formulation
------------------------
For each binary measurement i, the model outputs an observable ``y_i(θ)``.
Two inner parameters are estimated per group of measurements:

- ``α_g`` — baseline death propensity for group g (e.g. cell line)
- ``β``   — shared logistic slope

Death probability:   ``p_i = σ(α_{g_i} + β · y_i)``
Likelihood:          ``ℓ = Σ_i [z_i log p_i + (1−z_i) log(1−p_i)]``

The inner optimization over (α, β) is a convex logistic regression, solved
numerically via scipy L-BFGS-B.  The outer gradient is obtained via the
envelope theorem:

    ∂(−ℓ)/∂θ_k = −Σ_i (z_i − p*_i) · β* · sy_i_k

where ``sy_i_k = ∂y_i/∂θ_k`` comes from AMICI forward sensitivities.

PEtab representation
--------------------
Binary measurements are identified by ``measurementType = "BINARY"`` in
the measurements table.  The ``observableParameters`` column carries two
semicolon-separated IDs::

    observableParameters = "alpha_group_id;beta_id"

Position 0 → α parameter (group baseline)
Position 1 → β parameter (shared slope)

The corresponding entries in ``parameters.tsv`` should have ``estimate = 0``
(they are not outer parameters; the binary submodule estimates them as inner
parameters).
"""

from .calculator import BinaryAmiciCalculator
from .parameter import BinaryInnerParameter, BinaryInnerParameterType
from .problem import BinaryInnerProblem
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
]
