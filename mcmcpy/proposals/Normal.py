from scipy.stats import norm

class Normal:

    def __init__(self, var):

        self.var = var

    def __call__(self, chain):
        return norm(loc=chain.current_state, scale=self.var).rvs()

    def pdf(self, x1, x2):
        return norm(loc=x2, scale=self.var).pdf(x1)

