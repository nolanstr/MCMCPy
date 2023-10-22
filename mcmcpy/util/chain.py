import numpy as np


class Chain:

    def __init__(self, x):
        self.current_state = x
        self.accepted_states = np.array([x])
        self.rejected_states = np.array([x])
        self.accepted_state_count = 0

    def accept_state(self, x):
        self.accepted_state_count += 1
        self.rejected_states = np.append(self.rejected_states,
                                            self.current_state)
        self.current_state = x
        self.accepted_states = np.append(self.accepted_states, x)
    
    def reject_state(self, x):
        self.accepted_states = np.append(self.accepted_states,
                                            self.current_state)
        self.rejected_states = np.append(self.rejected_states, x)

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

    def __getitem__(self, index):
        return self.chains[index]
