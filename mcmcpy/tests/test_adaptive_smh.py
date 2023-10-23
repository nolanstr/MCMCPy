import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from mcmcpy.samplers.metropolis_hastings import SingleChainMetropolisHastings
from mcmcpy.proposals.normal import AdaptiveNormal

if __name__ == "__main__":

    x = 1
    density = lambda X: norm(loc=0, scale=1).pdf(X)
    proposal = AdaptiveNormal(var=1)
    SMH = SingleChainMetropolisHastings(density, proposal)
    SMH.sample(x, steps=10000)
    print(f"Accepted Rate: {SMH.chain.accepted_state_count/SMH.steps}")    
    t = np.arange(SMH.steps)
    fig, ax = plt.subplots()
    ax.plot(t, SMH.chain.rejected_states, label="Rejected")
    ax.plot(t, SMH.chain.accepted_states, label="Accepted")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.legend()

    plt.show()

    import pdb;pdb.set_trace()
