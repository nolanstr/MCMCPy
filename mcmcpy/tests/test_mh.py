import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from mcmcpy.samplers.metropolis_hastings import MetropolisHastings
from mcmcpy.proposals.normal import Normal

if __name__ == "__main__":

    x = [1,2,3]
    density = lambda X: norm(loc=0, scale=1).pdf(X)
    proposal = Normal(var=1)
    MH = MetropolisHastings(density, proposal)
    MH.sample(x, steps=1000)
    for i in range(len(x)):
        print(f"Accepted Rate for chain {i}: {MH.chains[i].accepted_state_count/MH.steps}")    
    t = np.arange(MH.steps)
    fig, ax = plt.subplots()
    for i in range(len(x)):
        ax.plot(t, MH.chains[i].rejected_states[:,0], color=plt.cm.tab10(i),
                label=f"Chain {i}")
        ax.plot(t, MH.chains[i].accepted_states[:,0], color=plt.cm.tab10(i),
                                                linestyle="--", alpha=0.3)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.legend()

    plt.show()

    import pdb;pdb.set_trace()
