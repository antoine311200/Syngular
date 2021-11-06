from __future__ import annotations

import numpy as np
import nptyping as ntp
import pandas as pd

import math
import time


from typing import NamedTuple, Union, Callable, Type
from progressbar.progressbar import ProgressBar

from sklearn.model_selection import ParameterGrid
from scipy.stats import gamma, beta, uniform

from syngular.utils.benchmark import Timer

class Parameter(NamedTuple):
    name: str
    call: Union[Callable[..., float], list[float], ntp.NDArray[float]]

print(ntp)

class Dataset:

    def __init__(self, increment, params: list[Parameter]) -> None:
        self.increment = increment
        self.percentiles = pd.Series(np.linspace(0, 0.99, self.increment))

        self.params = params
        self.grid = {}
        self.pregrid = {}

        self.bar = ProgressBar()
        self.dataframe = pd.DataFrame()

    @Timer.wrapper
    def generate(self):
        for p in self.params:
            print(p.name)
            if isinstance(p.call, list[float].__origin__) or isinstance(p.call, ntp.NDArray[float].__origin__):
                # if len(p.call) == self.increment:
                self.pregrid[p.name] = p.call
                # else:
                    # raise IndexError("Parameter list of values must be the same size of the increment")
            elif isinstance(p.call, Callable[..., float].__origin__):
                self.pregrid[p.name] = self.percentiles.apply(p.call)
            else:
                raise TypeError("Parameter call must be a float function or a list of floats")

        self.grid = ParameterGrid(self.pregrid)

        for params in self.bar(self.grid):
            self.dataframe = self.dataframe.append(pd.Series(params), ignore_index=True)

        # print(self.dataframe.head())

        return self

    def add_column(self, name, col):
        self.dataframe[name] = col

    def __add__(self, dataset: Dataset):
        return Dataset(self.increment + dataset.increment, self.params + dataset.params)

    @staticmethod
    def empty():
        return Dataset(0, [])

    @staticmethod
    def concatenate(*datasets: list[Type[Dataset]]):
        dt_list = list(datasets)
        dt_concat = Dataset.empty()

        while len(dt_list) > 0:
            dt = dt_list.pop()
            dt_concat += dt
        return dt_concat

    def __str__(self):
        return self.dataframe.__repr__()



# dt = Dataset(2, [
#     Parameter(name = "S", call = lambda x : gamma.ppf(x, a=100, scale=1)),
#     Parameter(name = "K", call = lambda x : uniform.ppf(x, 50, 200)),
#     Parameter(name = "R", call = lambda x : uniform.ppf(x, 0.01, 0.18)),
#     Parameter(name = "D", call = lambda x : uniform.ppf(x, 0.01, 0.18)),
#     Parameter(name = "sigma", call = lambda x : (beta.ppf(x, a=2, b=5) + 0.001))
# ])

# dt2 = Dataset(2, [
#     Parameter(name = "S", call = lambda x : gamma.ppf(x, a=100, scale=1)),
#     Parameter(name = "K", call = lambda x : uniform.ppf(x, 50, 200)),
#     Parameter(name = "R", call = lambda x : uniform.ppf(x, 0.01, 0.18)),
#     Parameter(name = "D", call = lambda x : uniform.ppf(x, 0.01, 0.18)),
#     Parameter(name = "sigma", call = lambda x : (beta.ppf(x, a=2, b=5) + 0.001))
# ])

# dt3 = Dataset.concatenate(dt,dt2)
# dt3.generate()

# print(dt3)
