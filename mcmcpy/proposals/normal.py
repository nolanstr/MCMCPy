import numpy as np
from scipy.stats import norm, multivariate_normal as mvn

class Normal:

    def __init__(self, **kwargs):
        if "var" in kwargs.keys():
            cov = np.array([[kwargs[list(kwargs.keys())[0]]]])
        else:
            cov = kwargs[list(kwargs.keys())[0]]
        self.cov = cov

    def __call__(self, chain):
        return mvn(mean=chain.current_state, cov=self.cov).rvs()

    def pdf(self, x1, x2):
        return mvn(mean=x2, cov=self.cov).pdf(x1)

class AdaptiveNormal:
    """
    Adaptive proposal from Haario et al.
    """
    def __init__(self, **kwargs):
        
        if "var" in kwargs.keys():
            var = kwargs[list(kwargs.keys())[0]]
            if isinstance(var, np.ndarray):
                if var.ndim==2:
                    cov = var
                elif var.ndim==1:
                    cov = np.array(var).reshape((1,1))
                else:
                    raise ValueError(
                            "Variance dimensions incorrect")

        elif "cov" in kwargs.keys():
            cov = kwargs[list(kwargs.keys())[0]]
            if not isinstance(cov, np.ndarray):
                raise ValueError(
                        "Covariance should be nxn matrix")
        else:
                raise ValueError(
                        "Requires either 'var' or 'cov' argument")
        self.cov = cov
        self._epsilon = 1e-6

    def __call__(self, chain):
        d = chain.current_state.shape[0]
        if chain.accepted_state_count > 0:
            cov = chain.compute_within_chain_covariance()
            if np.any(np.isinf(cov)) or np.any(np.isnan(cov)):
                cov = self.cov
            self.cov = cov
            self.cov = (((2.38**2)/d) * self.cov) + \
                    (self._epsilon*np.eye(d))
        return mvn(mean=chain.current_state, cov=self.cov).rvs()

    def pdf(self, x1, x2):
        return mvn(mean=x2, cov=self.cov).pdf(x1)


