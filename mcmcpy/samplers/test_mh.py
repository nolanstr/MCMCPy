import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from mcmcpy.samplers.metropolis_hastings import MetropolisHastings
from mcmcpy.proposals.Normal import Normal

if __name__ == "__main__":

    x = np.array([1])
    density = lambda X: norm(loc=0, scale=1).pdf(X)
    proposal = Normal(var=1)
    MH = MetropolisHastings(density, proposal)
    MH.sample(x, steps=10000)
    print(f"Accepted Rate: {MH.chains[0].accepted_state_count/MH.steps}")    
    t = np.arange(MH.steps)
    fig, ax = plt.subplots()
    ax.plot(t, MH.chains[0].rejected_states, label="Rejected")
    ax.plot(t, MH.chains[0].accepted_states, label="Accepted")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.legend()

    plt.show()

    import pdb;pdb.set_trace()
