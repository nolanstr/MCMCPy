import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

from mcmcpy.samplers.metropolis_hastings import SingleChainMetropolisHastings
from mcmcpy.proposals.normal import AdaptiveNormal

if __name__ == "__main__":

    # Initialize first sample and mean and covariance of proposal distribution
    x = np.array([0,0])
    
    mu1 = np.zeros(2)
    cov1 = np.eye(2)
    mu2 = np.array([2,2])
    var2_1 = pow(0.3,2)
    var2_2 = pow(0.3,2)
    rho2 = 0.8
    cov2 = np.array([[var2_1, rho2*np.sqrt(var2_1*var2_2)],
                     [rho2*np.sqrt(var2_1*var2_2), var2_2]])
    dist1 = mvn(mean=mu1, cov=cov1)
    dist2 = mvn(mean=mu2, cov=cov2)

    density = lambda X: (2/3) * dist1.pdf(X) + (1/3)*dist2.pdf(X)
    proposal = AdaptiveNormal(var=1)
    SMH = SingleChainMetropolisHastings(density, proposal)
    SMH.sample(x, steps=10000)
    print(f"Accepted Rate: {SMH.chain.accepted_state_count/SMH.steps}")    
    t = np.arange(SMH.steps)
    fig, axs = plt.subplots(2)
    axs[0].set_title("Trace for x samples")
    axs[0].plot(t, SMH.chain.rejected_states[:,0], label="Rejected")
    axs[0].plot(t, SMH.chain.accepted_states[:,0], label="Accepted")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("x")
    axs[0].legend()

    axs[1].set_title("Trace for y samples")
    axs[1].plot(t, SMH.chain.rejected_states[:,1], label="Rejected")
    axs[1].plot(t, SMH.chain.accepted_states[:,1], label="Accepted")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("x")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
     
    ess = SMH.chain.estimate_effective_sample_size()

    import pdb;pdb.set_trace()
