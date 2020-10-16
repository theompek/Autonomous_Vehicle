"""
This module represent the borders of a base path.

"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from time import sleep
import math
import numpy as np

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================


class Border:
    """
        The class includes all the functions to retrieve information about borders of a driving lane,
        includes all the necessary information of the position and the type of the boarder.
    """

    def __init__(self):
        self.x = []
        self.y = []
        self.lane_marking = []
        self.lane_change = []

    def length(self):
        return len(self.x)

    def getFromTo(self, _from, _to):
        border = Border()
        border.x = self.x[_from:_to]
        border.y = self.y[_from:_to]
        border.lane_change = self.lane_change[_from:_to]

        return border



