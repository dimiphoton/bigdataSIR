import pymc as pm

def define_priors(model):
    """
    Define the prior distributions for the SEIR model parameters.

    Parameters:
    - model: A PyMC model instance.

    Returns:
    - priors: A dictionary containing the prior distributions.
    """
    with model:
        beta = pm.Normal('beta', mu=0.3, sigma=0.1)
        sigma = pm.Normal('sigma', mu=0.2, sigma=0.1)
        p_sym = pm.Uniform('p_sym', lower=0.5, upper=0.7)
        gamma = pm.Normal('gamma', mu=0.1, sigma=0.1)
        hosp_rate = pm.Normal('hosp_rate', mu=0.05, sigma=0.01)
        mu = pm.Normal('mu', mu=0.02, sigma=0.01)
        p_test = pm.Uniform('p_test', lower=0.1, upper=1.0)

    priors = {
        'beta': beta,
        'sigma': sigma,
        'p_sym': p_sym,
        'gamma': gamma,
        'hosp_rate': hosp_rate,
        'mu': mu,
        'p_test': p_test
    }

    return priors
