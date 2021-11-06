import matplotlib.pyplot as plt

import numpy as np
import math

class StochasticProcess:
    name: str

    def simulate(self):
        pass

class RandomWalk1D(StochasticProcess):
    def __init__(self, iteration: float, origin: float = 0, step_set: list[float] = [-1, 0, 1]) -> None:
        self.iteration = iteration
        self.step_set = step_set
        self.origin = [origin]

        self.simulate()

    def simulate(self):
        self.steps = np.random.choice(a = self.step_set, size = (self.iteration, 1))
        print(self.steps, self.origin)
        self.path = np.concatenate([[self.origin], self.steps]).cumsum(0)

class BrownianMotion(StochasticProcess):
    def __init__(self, origin: float, dt: float, time: float) -> None:
        self.origin = origin
        self.current = self.origin
        self.time = time
        self.initial_time = time
        self.dt = dt
        self.iteration = self.time / self.dt

        self.path = [self.origin]

        self.simulate()
    
    def simulate(self):
        while(self.time - self.dt > 0):
            dW = np.random.normal(0, math.sqrt(self.dt))
            value = self.current + dW
            self.current = value

            self.path.append(value)

            self.time -= self.dt
    
    def pdf(self, x: float, t: float) -> float:
        if t != None:
            timeline = np.linspace(self.dt, self.initial_time, int(self.iteration)-1)
            return 1 / np.sqrt(2 * np.pi * timeline) * np.exp(-x**2 / (2*timeline))
        else:
            return 1 / np.sqrt(2 * np.pi * t) * np.exp(-x**2 / (2*t))

WienerProcess = BrownianMotion

class GeometricBrownianMotion(StochasticProcess):

    def __init__(self, initial_price: float, drift: float, volatility: float, dt: float, time: float) -> None:
        self.current_price = initial_price
        self.initial_price = initial_price

        self.name = "Geometric Brownian Motion"

        self.drift = drift
        self.volatility = volatility
        self.dt = dt
        self.time = time

        self.prices = []

        self.simulate()

    # Simulate the following equation : 
    # dYt/Yt = mu.dt + sigma.dWt with dWt ～　N(0, sqrt(dt))
    # where mu is the drift, sigma the volatility of the motion
    def simulate(self):
        while(self.time - self.dt > 0):
            dWt = np.random.normal(0, math.sqrt(self.dt))
            dYt = self.drift * self.dt + self.volatility * dWt

            self.current_price += dYt
            self.prices.append(self.current_price)

            self.time -= self.dt

plt.plot(BrownianMotion(origin=0, dt=1/365, time=1).pdf(0.5))
for i in range(4):
    # plt.plot(GeometricBrownianMotion(100, 0.08, 0.1, 1/365, 1).prices)
    # plt.plot(RandomWalk1D(origin=0, iteration=10000).path)
    # plt.plot(BrownianMotion(origin=0, dt=1/365, time=1).path)
    pass
plt.show()