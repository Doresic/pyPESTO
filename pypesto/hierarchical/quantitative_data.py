import logging
from typing import Any

logger = logging.getLogger(__name__)


class QuantitativeData:

    SCALING = 'scaling'
    OFFSET = 'offset'
    SIGMA = 'sigma'
    OPTIMALSCALING = 'qualitative_scaling'

    def __init__(self,
                 id: str,
                 condition: str,
                 measurement: int ,
                 group: int = None):
        """
        Quantitative data
        ----------
        id: str
            Id of the parameter.
        boring_val: float
            Value to be used when the parameter is not present (in particular
            to simulate unscaled observables).
        """
        self.id: str = id
        self.measurement: int = measurement
        self.condition= condition

        if group is None:
            raise ValueError("No Parameter group provided.")
        self.group = group
