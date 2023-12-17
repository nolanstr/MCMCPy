import sys;sys.path.append("../../../")
import numpy as np
from scipy.stats import multivariate_normal as mvn

from mcmcpy.subset_simulation.subset_simulation import SubsetSimulation
from mcmcpy.proposals.normal import AdaptiveNormal

density = mvn(mean=np.zeros(2), cov=np.eye(2))
performance_function = lambda X: 4*np.sqrt(2) - X[:,0] - X[:,1]
proposal = AdaptiveNormal(cov=np.eye(2))
number_of_chains = 10
number_of_samples = 100

SubsetSimulator = SubsetSimulation(density, 
                                   proposal, 
                                   performance_function, 
                                   number_of_chains, 
                                   number_of_samples)
failure_probability, initial_samples, retained_samples = SubsetSimulator()
