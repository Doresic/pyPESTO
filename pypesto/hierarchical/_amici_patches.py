"""Runtime monkeypatch of ``amici.petab.conditions`` for the Frohlich model.

Previously this lived as a hand-edit to ``amici/petab/conditions.py`` *inside the
venv* -- undiffable, not version-controlled, and silently lost on every reinstall.
The fork's hierarchical calculators call
``fill_in_parameters(..., recalc_plists_and_scales=...)``, a keyword argument stock
AMICI does not have, so a clean clone + stock AMICI raised ``TypeError`` on the first
gradient evaluation.

This module reinstates that patch at runtime: it replaces
``amici.petab.conditions.fill_in_parameters`` and
``...fill_in_parameters_for_condition`` with versions that accept
``recalc_plists_and_scales`` and, when it is ``False``, *skip* recomputing
``edata.plist``/``edata.pscale`` (a performance win -- the plists are identical across
conditions for this model, so they are computed once and reused).

The two functions below are vendored verbatim from AMICI 0.34.2's ``conditions.py``,
with only the ``recalc_plists_and_scales`` kwarg + guard added. They are therefore
coupled to that AMICI version *by design*: the version check below fails loudly if
AMICI ever changes, which is exactly the property we lost with the silent venv edit.

Importing this module applies the patch (idempotently). It is imported from
``inner_calculator_collector.py`` and ``relative/calculator.py`` -- before their own
import of ``fill_in_parameters`` -- so the patch is always in place before use.
"""

from __future__ import annotations

import warnings

import amici
import amici.petab.conditions as _cond
import numpy as np
from amici.petab.parameter_mapping import (
    petab_to_amici_scale,
    scale_parameters_dict,
    unscale_parameters_dict,
)

_SUPPORTED_AMICI_VERSION = "0.34.2"


def fill_in_parameters(
    edatas,
    problem_parameters,
    scaled_parameters,
    parameter_mapping,
    amici_model,
    warn_unused: bool = True,
    recalc_plists_and_scales: bool = True,
) -> None:
    """Fill fixed and dynamic parameters into the edatas (in-place).

    Vendored from AMICI 0.34.2 with the ``recalc_plists_and_scales`` kwarg added
    and forwarded to :func:`fill_in_parameters_for_condition`.
    """
    if warn_unused and (
        unused_parameters := (
            set(problem_parameters.keys()) - parameter_mapping.free_symbols
        )
    ):
        warnings.warn(
            "The following problem parameters were not used: "
            + str(unused_parameters),
            RuntimeWarning,
            stacklevel=2,
        )

    for edata, mapping_for_condition in zip(
        edatas, parameter_mapping, strict=True
    ):
        fill_in_parameters_for_condition(
            edata,
            problem_parameters,
            scaled_parameters,
            mapping_for_condition,
            amici_model,
            recalc_plists_and_scales=recalc_plists_and_scales,
        )


def fill_in_parameters_for_condition(
    edata,
    problem_parameters,
    scaled_parameters,
    parameter_mapping,
    amici_model,
    recalc_plists_and_scales: bool = True,
) -> None:
    """Fill fixed and dynamic parameters into the edata for condition (in-place).

    Vendored from AMICI 0.34.2. The only change is the ``recalc_plists_and_scales``
    kwarg: when ``False`` the ``scales``/``plist``/``pscale`` block is skipped, so
    those (condition-invariant for this model) are computed once and reused.
    """
    map_sim_var = parameter_mapping.map_sim_var
    scale_map_sim_var = parameter_mapping.scale_map_sim_var
    map_preeq_fix = parameter_mapping.map_preeq_fix
    scale_map_preeq_fix = parameter_mapping.scale_map_preeq_fix
    map_sim_fix = parameter_mapping.map_sim_fix
    scale_map_sim_fix = parameter_mapping.scale_map_sim_fix

    # Parameter mapping may contain parameter_ids as values, these *must*
    # be replaced

    def _get_par(model_par, value, mapping):
        """Replace parameter IDs in mapping dicts by values from
        problem_parameters where necessary"""
        if isinstance(value, str):
            try:
                # estimated parameter
                return problem_parameters[value]
            except KeyError:
                # condition table overrides must have been handled already,
                # e.g. by the PEtab parameter mapping, but parameters from
                # InitialAssignments may still be present.
                if (mapped_value := mapping[value]) == model_par:
                    # prevent infinite recursion
                    raise
                return _get_par(value, mapped_value, mapping)

        try:
            # user-provided
            return problem_parameters[model_par]
        except KeyError:
            pass

        # prevent nan-propagation in derivative
        if np.isnan(value):
            return 0.0

        # constant value
        return value

    map_preeq_fix = {
        key: _get_par(key, val, map_preeq_fix)
        for key, val in map_preeq_fix.items()
    }
    map_sim_fix = {
        key: _get_par(key, val, map_sim_fix)
        for key, val in map_sim_fix.items()
    }
    map_sim_fix_var = map_sim_fix | map_sim_var
    map_sim_var = {
        key: _get_par(key, val, map_sim_fix_var)
        for key, val in map_sim_var.items()
    }

    # If necessary, (un)scale parameters
    if scaled_parameters:
        unscale_parameters_dict(map_preeq_fix, scale_map_preeq_fix)
        unscale_parameters_dict(map_sim_fix, scale_map_sim_fix)
    if not scaled_parameters:
        # We scale all parameters to the scale they are estimated on, and pass
        # that information to amici via edata.{parameters,pscale}.
        # The scaling is necessary to obtain correct derivatives.
        scale_parameters_dict(map_sim_var, scale_map_sim_var)
        # We can skip preequilibration parameters, because they are identical
        # with simulation parameters, and only the latter are used from here
        # on.

    ##########################################################################
    # variable parameters and parameter scale

    # parameter list from mapping dict
    parameters = [
        map_sim_var[par_id] for par_id in amici_model.getParameterIds()
    ]

    if parameters:
        edata.parameters = np.asarray(parameters, dtype=float)

    if recalc_plists_and_scales:
        # scales list from mapping dict
        scales = [
            petab_to_amici_scale(scale_map_sim_var[par_id])
            for par_id in amici_model.getParameterIds()
        ]

        # plist
        plist = [
            ip
            for ip, par_id in enumerate(amici_model.getParameterIds())
            if isinstance(parameter_mapping.map_sim_var[par_id], str)
        ]

        if scales:
            edata.pscale = amici.parameterScalingFromIntVector(scales)

        if plist:
            edata.plist = plist

    ##########################################################################
    # fixed parameters preequilibration
    if map_preeq_fix:
        fixed_pars_preeq = [
            map_preeq_fix[par_id]
            for par_id in amici_model.getFixedParameterIds()
        ]
        edata.fixedParametersPreequilibration = fixed_pars_preeq

    ##########################################################################
    # fixed parameters simulation
    if map_sim_fix:
        fixed_pars_sim = [
            map_sim_fix[par_id]
            for par_id in amici_model.getFixedParameterIds()
        ]
        edata.fixedParameters = fixed_pars_sim


def apply() -> None:
    """Install the patched functions onto ``amici.petab.conditions`` (idempotent)."""
    if getattr(_cond, "_frohlich_amici_patched", False):
        return

    if amici.__version__ != _SUPPORTED_AMICI_VERSION:
        raise RuntimeError(
            "frohlich_cancer pyPESTO fork vendors the AMICI "
            f"{_SUPPORTED_AMICI_VERSION} 'fill_in_parameters' bodies, but the "
            f"installed AMICI is {amici.__version__}. Re-vendor "
            "pypesto/hierarchical/_amici_patches.py from this AMICI version's "
            "conditions.py before continuing."
        )

    _cond.fill_in_parameters = fill_in_parameters
    _cond.fill_in_parameters_for_condition = fill_in_parameters_for_condition
    _cond._frohlich_amici_patched = True


# Applied on import (the import sites guarantee this runs before any caller's
# own `from amici.petab.conditions import fill_in_parameters`).
apply()
