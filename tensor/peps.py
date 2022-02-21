from __future__ import annotations
from typing import Tuple, List, Union

import numpy as np
from pandas import array
from scipy import linalg

from opt_einsum import contract

class PEPState:

    """
    Example: a 3x3 PEPS
           |   |   |
        ---O---O---O---
         / | / | / |
        ---O---O---O---
         / | / | / |
        ---O---O---O---
         / | / | / |

    Convention:

               | (t)
        (l) ---O--- (r)  ==> (l, t, i, b, r)
        (i)  / | (b)

    """

    def __init__(self, tensor, bond_matrix: List[List[int]]) -> None:
        self.parameters_number = 0
        self.real_parameters_number = 0

        if tensor is not None:
            self.tensor_shape = tensor.shape
            self.bond_matrix = np.array(bond_matrix)

            self.width = bond_matrix.shape[0]
            self.height = bond_matrix.shape[1]

            self.order = len(self.tensor_shape)
            self.real_parameters_number = np.prod(self.tensor_shape)
            
            self.sites_number = len(self.bond_shape)+1
            self.sites = [None] * self.sites_number

            if self.bond_matrix != ():
                self.shape = []
                for w in self.width:
                    for h in self.height:
                        shp = (0,0,0,0)
                        if w == 0:
                            shp[0] = 1
                        elif w == self.width-1:
                            shp[3] = 1

    @staticmethod
    def ground(width: int, height: int) -> PEPState:
        pass
