# 📁 seir_model.py

import numpy as np

class SEIRSimulator:
    def __init__(self, N, params, stochastic=True):
        self.N = N
        self.params = params
        self.stochastic = stochastic

    def _transition(self, n, p):
        if self.stochastic:
            return np.random.binomial(n, p)
        else:
            return round(n * p)

    def step(self, state):
        S, E, I, H, ICU, R, D = state
        p = self.params

        new_exposed = self._transition(S, p["beta"] * I / self.N)
        new_infectious = self._transition(E, p["sigma"])
        new_recovered_I = self._transition(I, p["gamma_I"])
        new_hospitalized = self._transition(I - new_recovered_I, p["h_rate"])

        new_recovered_H = self._transition(H, p["gamma_H"])
        new_icu = self._transition(H - new_recovered_H, p["icu_rate"])

        new_recovered_ICU = self._transition(ICU, p["gamma_ICU"])
        new_dead = self._transition(ICU - new_recovered_ICU, p["mu"])

        S -= new_exposed
        E += new_exposed - new_infectious
        I += new_infectious - new_recovered_I - new_hospitalized
        H += new_hospitalized - new_recovered_H - new_icu
        ICU += new_icu - new_recovered_ICU - new_dead
        R += new_recovered_I + new_recovered_H + new_recovered_ICU
        D += new_dead

        return [S, E, I, H, ICU, R, D]

    def simulate(self, initial_state, days):
        history = [initial_state]
        for _ in range(days):
            next_state = self.step(history[-1])
            history.append(next_state)
        return np.array(history)
