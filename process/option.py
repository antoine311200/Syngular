import numpy as np

from progressbar import ProgressBar
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import uniform
from scipy.stats import norm

import math
import time

from typing import Any, Optional, Tuple, NamedTuple

class Law(NamedTuple):
    name: str
    min: float
    max: float

normal = Law(name = 'normal', min = 0, max = 1)

class Model:
    def __init__(self, iterations) -> None:
        self.iterations = iterations

    def monte_carlo(self, law: Law):
        if law.name == 'normal':
            return np.random.normal(law.min, law.max, [1, self.iterations])

class Pricing:
    def __init__(self) -> None:
        pass

class BlackScholes(Pricing):
    def __init__(self, F, K, t, r, sigma) -> None:
        # self.S = S
        self.K = K
        self.t = t
        self.r = r
        self.sigma = sigma

        self.deflater = np.exp(-self.r*self.t)
        self.F = F

        self.N = norm.cdf

    def d1(self) -> float:
        return (np.log(self.F / self.K) + (np.power(self.sigma, 2) * self.t / 2.0 )) / (self.sigma * np.sqrt(self.t))

    def d2(self) -> float:
        return (np.log(self.F / self.K) - (np.power(self.sigma, 2) * self.t / 2.0 )) / (self.sigma * np.sqrt(self.t))

    def put(self) -> float:
        return self.deflater * (-self.F * self.N(-self.d1()) + self.K * self.N(-self.d2()))

    def call(self) -> float:
        return self.deflater * (self.F * self.N(self.d1()) - self.K * self.N(self.d2()))


class BlackScholesMerton(BlackScholes):
    def __init__(self, S, K, t, r, q, sigma) -> None:
        super().__init__(S * np.exp((r-q)*t), K, t, r, sigma)
        self.S = S
        self.q = q

class Option(Model):

    def __init__(self, pricing: Pricing) -> None:
        super().__init__()

        self.pricing = pricing

    def call(self):
        self.pricing.call()

    def put(self):
        self.pricing.put()


if __name__ == "__main__":
    
    S = 100
    K = 95
    q = .05
    t = 0.5
    r = 0.1
    sigma = 0.2
    p_published_value = 2.4648
    p_calc = BlackScholesMerton(S, K, t, r, q, sigma).put()
    print(p_calc)
    print(abs(p_published_value - p_calc) < 0.0001)