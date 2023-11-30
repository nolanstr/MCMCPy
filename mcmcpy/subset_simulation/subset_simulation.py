import numpy as np

from mcmcpy.samplers.metropolis_hastings import MetropolisHastings

class SubsetSimulation:

    def __init__(self, density, proposal, performance_function,
                    number_of_chains, number_of_samples): 

        self.density = density
        self.proposal = proposal

        self.performance_function = performance_function
        self.N = number_of_samples
        self.N_c = number_of_chains
        self.N_s = int(self.N/self.N_c)
        self.target_cond_prob = self.N_c/self.N 

    def __call__(self, number_of_failure_events=10):

        initial_samples = self.density.rvs(self.N)
        scores = self.performance_function(initial_samples)
        x_i = initial_samples[np.argsort(scores)[:self.N_c],:]
        retained_samples = [x_i.copy()]

        for i in range(number_of_failure_events):
            self.quantile_value = np.max(self.performance_function(x_i))
            sampler = MetropolisHastings(self.quantile_density,
                                         self.proposal)
            sampler.sample(x_i, steps=self.N_s)
            x_0 = np.vstack([chain.accepted_states \
                            for chain in sampler.chains])            
            scores = self.performance_function(x_0)
            x_i = x_0[np.argsort(scores)[:self.N_c],:]
            retained_samples.append(x_i.copy())
            final_prob = np.sum(scores<0)/scores.shape[0]
            if final_prob>self.target_cond_prob: 
                number_of_failure_events = i+1
                break
        
        failure_prob = np.power(self.target_cond_prob, 
                    number_of_failure_events) * final_prob

        return failure_prob, initial_samples, retained_samples

    def quantile_density(self, X):

        probability = self.density.pdf(X)
        if self.performance_function(
                np.expand_dims(X, axis=1).T) < self.quantile_value:
            return probability
        else:
            return 0
