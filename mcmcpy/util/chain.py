import numpy as np


class Chain:

    def __init__(self, x):
        x = self.check_x_value(x)
        self.current_state = x
        self.accepted_states = x
        self.rejected_states = x
        self.accepted_state_count = 0

    def accept_state(self, x):
        x = self.check_x_value(x)
        self.accepted_state_count += 1
        self.rejected_states = np.vstack((self.rejected_states,
                                            self.current_state))
        self.current_state = x
        self.accepted_states = np.vstack((self.accepted_states, x))
    
    def reject_state(self, x):
        x = self.check_x_value(x)
        self.accepted_states = np.vstack((self.accepted_states,
                                            self.current_state))
        self.rejected_states = np.vstack((self.rejected_states, x))

    def check_x_value(self, x):
        if not isinstance(x, np.ndarray):
            return np.array([x])
        else:
            return x.flatten()

    def compute_within_chain_mean(self):
        self.chain_mean = np.mean(self.accepted_states, axis=0)
        return self.chain_mean

    def compute_within_chain_variance(self):
        self.chain_variance = np.var(self.accepted_states, axis=0)
        return self.chain_variance

    def compute_within_chain_covariance(self):
        self.chain_covariance = np.cov(self.accepted_states.T) 
        return self.chain_covariance
    
    def compute_chain_correlations(self):
        self.chain_correlations = []
        for i in range(self.accepted_states.shape[1]):
            self.chain_correlations.append(np.correlate(
                                                self.accepted_states[:,i],
                                                self.accepted_states[:,i], 
                                                mode="full"))
        self.chain_correlations = np.array(self.chain_correlations).T
        #maybe use scipy signal correlation function
        return self.chain_correlations

    def estimate_effective_sample_size(self, burn_in=0):
        n = self.accepted_state_count
        self.compute_chain_correlations()
        import pdb;pdb.set_trace()
        self.ess = n / (1 + (2*np.sum(self.chain_correlations, axis=0)/n))
        return self.ess

class MultipleChains:

    def __init__(self, xs):
        """
        Add more functionality later for computing between-chain 
        correlations?
        """
        xs = self._check_args(xs)
        self.chains = [Chain(x) for x in xs]
        self.number_of_chains = len(self.chains)
    
    def _check_args(self, xs):

        assert(isinstance(xs, np.ndarray) or isinstance(xs, list),
                "Initial samples must be numpy array or list")
        if isinstance(xs, np.ndarray):
            return xs.flatten()
        else:
            return xs
    
    def estimate_effective_sample_size(self):
        M = self.number_of_chains
        N = chains[0].accepted_states.shape[0]
        chains_means = np.array([chain.compute_within_chain_mean() \
                                            for chain in chains])
        total_mean = np.mean(chains_means)
        B = (N/(M-1)) * np.sum(np.square(chains_means-total_mean))
        W = np.mean([chain.compute_within_chain_variance() \
                                    for chain in self.chains])
        total_variance
        

    def __getitem__(self, index):
        return self.chains[index]
