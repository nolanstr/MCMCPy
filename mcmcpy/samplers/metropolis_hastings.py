import numpy as np

from mcmcpy.util.chain import Chain, MultipleChains

class MetropolisHastings:

    def __init__(self, density, proposal):

        self.density = density
        self.proposal = proposal

    def sample(self, x, steps=100):
        
        self._check_args(x)
        self.chains = MultipleChains(x)
        self.number_of_chains = self.chains.number_of_chains
        self.steps = steps+1
        for step in range(steps):
            for chain in self.chains:
                self.perform_step(chain) 
    
    def perform_step(self, chain):

        x_proposal = self.proposal(chain)
        num = self.density(x_proposal) * self.proposal.pdf(
                chain.current_state, x_proposal)
        den = self.density(chain.current_state) * \
                self.proposal.pdf(x_proposal, chain.current_state)
        alpha = num/den
        if np.random.uniform()<alpha:
            chain.accept_state(x_proposal)
        else:
            chain.reject_state(x_proposal)

    def _check_args(self, x):

        assert(isinstance(x, np.ndarray) or isinstance(x, list)),\
                "initial sample must be numpy array or list"

class SingleChainMetropolisHastings:
    """
    Generalized single chain MCMC metropolis hastings code that allows for use
    of adaptive proposal.
    """

    def __init__(self, density, proposal):

        self.density = density
        self.proposal = proposal
        self.number_of_chains = 1 

    def sample(self, x, steps=100):
        
        self._check_args(x)
        self.chain = Chain(x)
        self.steps = steps+1
        for step in range(steps):
            self.perform_step(self.chain) 
    
    def perform_step(self, chain):

        x_proposal = self.proposal(chain)
        num = self.density(x_proposal) * self.proposal.pdf(
                chain.current_state, x_proposal)
        den = self.density(chain.current_state) * \
                self.proposal.pdf(x_proposal, chain.current_state)
        alpha = num/den
        if np.random.uniform()<alpha:
            chain.accept_state(x_proposal)
        else:
            chain.reject_state(x_proposal)

    def _check_args(self, x):

        assert(isinstance(x, float) or \
                isinstance(x, int)) or \
                isinstance(x, np.ndarray),\
                "initial sample must be float, int, or npndarray"
