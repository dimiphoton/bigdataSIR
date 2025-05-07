import numpy as np

class SEIRModel:
    def __init__(self, beta, sigma, p_sym, gamma, hosp_rate, mu, S0, E0, I0, R0, p_test, sens, spec, stochastic=False):
        self.beta = beta
        self.sigma = sigma
        self.p_sym = p_sym
        self.gamma = gamma
        self.hosp_rate = hosp_rate
        self.mu = mu
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.p_test = p_test
        self.sens = sens
        self.spec = spec
        self.stochastic = stochastic
        self.N = S0 + E0 + I0 + R0  # Total population size
        self.parameters = {
            'beta': beta,
            'sigma': sigma,
            'p_sym': p_sym,
            'gamma': gamma,
            'hosp_rate': hosp_rate,
            'mu': mu,
            'p_test': p_test,
            'sens': sens,
            'spec': spec
        }

    def derivatives(self, S, E, I, Sym, Asym, Hosp, R, D):
        if self.stochastic:
            beta = np.random.normal(self.beta, 0.01 * abs(self.beta))
            sigma = np.random.normal(self.sigma, 0.01 * abs(self.sigma))
            p_sym = np.clip(np.random.normal(self.p_sym, 0.01 * abs(self.p_sym)), 0, 1)
            gamma = np.random.normal(self.gamma, 0.01 * abs(self.gamma))
            hosp_rate = np.random.normal(self.hosp_rate, 0.01 * abs(self.hosp_rate))
            mu = np.random.normal(self.mu, 0.01 * abs(self.mu))
        else:
            beta, sigma, p_sym, gamma, hosp_rate, mu = self.beta, self.sigma, self.p_sym, self.gamma, self.hosp_rate, self.mu

        # Derivatives for the SEIR model
        dS = -beta * S * I / self.N
        dE = beta * S * I / self.N - sigma * E
        dI = sigma * E - I
        dSym = p_sym * I - hosp_rate * Sym - gamma * Sym
        dAsym = (1 - p_sym) * I - gamma * Asym
        dHosp = hosp_rate * Sym - mu * Hosp
        dR = gamma * (Sym + Asym) + (1 - mu) * Hosp
        dD = mu * Hosp

        return dS, dE, dI, dSym, dAsym, dHosp, dR, dD

    def step(self, state):
        # Use the derivatives method to get the rate of change for the state
        dS, dE, dI, dSym, dAsym, dHosp, dR, dD = self.derivatives(*state)

        # Update the state by adding the derivatives
        state = tuple(max(0, s + ds) for s, ds in zip(state, (dS, dE, dI, dSym, dAsym, dHosp, dR, dD)))

        return state
