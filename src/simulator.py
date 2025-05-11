# seir_model/simulator.py

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
            return n * p  # No rounding, supports float transitions

    def step(self, state):
        S, E, I, H, ICU, R, D = state
        p = self.params

        # Transition probabilities derived from average durations
        sigma = 1 / p["incubation"]
        gamma_I = 1 / p["symptomatic"]
        gamma_H = 1 / p["h_duration"]
        gamma_ICU = 1 / p["icu_duration"]

        # Transitions
        new_exposed = self._transition(S, p["beta"] * I / self.N)
        new_infectious = self._transition(E, sigma)

        new_recovered_I = self._transition(I, gamma_I * (1 - p["h_rate"]))
        new_hospitalized = self._transition(I, gamma_I * p["h_rate"])

        new_recovered_H = self._transition(H, gamma_H * (1 - p["icu_rate"]))
        new_icu = self._transition(H, gamma_H * p["icu_rate"])

        new_recovered_ICU = self._transition(ICU, gamma_ICU * (1 - p["mu"]))
        new_dead = self._transition(ICU, gamma_ICU * p["mu"])

        # Update compartments
        S -= new_exposed
        E += new_exposed - new_infectious
        I += new_infectious - new_recovered_I - new_hospitalized
        H += new_hospitalized - new_recovered_H - new_icu
        ICU += new_icu - new_recovered_ICU - new_dead
        R += new_recovered_I + new_recovered_H + new_recovered_ICU
        D += new_dead

        return [S, E, I, H, ICU, R, D]

    def simulate(self, initial_state, days):
        state = list(map(float, initial_state))  # force float for deterministic
        history = [state.copy()]

        for _ in range(days):
            state = self.step(state)
            history.append(state.copy())

        return np.array(history)
