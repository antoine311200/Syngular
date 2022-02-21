from __future__ import annotations
from typing import Tuple, List, Union

import numpy as np
from scipy import linalg

from opt_einsum import contract

class Index:
    name: str
    dim: int

GeneralizedSite = dict[str, list[Index]]

# class GeneralizedSite:

#     def __init__(self, groups: Group) -> None:
#         self.groups = groups

#     def __getitem__(self, index: str):
#         return self.groups[]