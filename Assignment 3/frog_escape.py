import sys
sys.path.append("../") # to access code from rl.distribution, etc.

from dataclasses import dataclass
import numpy as np
from typing import Mapping, Dict, Tuple
from rl.distribution import Categorical, FiniteDistribution, Distribution
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess 
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
import matplotlib.pyplot as plt
import itertools

np.set_printoptions(formatter={'float': lambda x: "{0:.3f}".format(x)})


@dataclass(frozen = True)
class Position:
	position : int


ActionMapping = Mapping[Position, Mapping[str, Categorical[Tuple[Position, float]]]]


class FrogEscape(FiniteMarkovDecisionProcess[Position, str]):
	def __init__(self, length : int):
		self.length = length # resembles n in the problem
		super().__init__(self.get_action_transition_reward_map())

	def get_action_transition_reward_map(self) -> ActionMapping:
		d : Dict[Position, Dict[str, Categorical[Tuple[Position, float]]]] = {}

		for state in range(1, self.length):
			state = Position(position = state)

			inner_dict : Dict[str, Categorical[Tuple[Position, float]]] = {}

			# if the frog croaks sound A...
			dict_A = {}

			dict_A[(Position(position = state.position - 1), 0.0)] = state.position / self.length
			if state.position + 1 == self.length: # if the frog escapes, reward = 1.0, else 0.0
				dict_A[(Position(position = self.length), 1.0)] = 1 - state.position / self.length

			else:
				dict_A[(Position(position = state.position + 1), 0.0)] = 1 - state.position / self.length

			inner_dict['A'] = Categorical(dict_A)

			# if the frog croaks sound B... if the frog escapes, reward = 1.0, else 0.0
			dict_B = {(Position(position = i), 0.0) : 1. / self.length for i in range(self.length) if i != state.position}
			dict_B[(Position(position = self.length), 1.0)] = 1. / self.length
			inner_dict['B'] = Categorical(dict_B)

			d[state] = inner_dict

		return d



if __name__ == '__main__':
	# now i plot the optimal escap-probability as a function of the states for n = 3, 6 and 9
	lengths = range(3, 10, 3)
	fig, axes = plt.subplots(len(lengths), figsize = (10, 7))
	for j, length in enumerate(lengths):
		puzzle : FiniteMarkovDecisionProcess[Position, str] = FrogEscape(length = length)

		# generate all possible combinations of actions for a given position
		outcomes = [[(pos, act) for act in ['A', 'B']] for pos in range(1, length)]
		optimal_policy = None # will store the optimal policy
		optimal_value = None # will store the optimal value function
		max_val = - np.inf # will be used to find the optimal value function

		# then form the cartesian product of all those lists
		for i in itertools.product(*outcomes):
			my_dict : Dict[Position, str] = {Position(position = pos) : act for pos, act in i}
			policy : FiniteDeterministicPolicy[Position, str] = FiniteDeterministicPolicy(my_dict)
			implied_mrp : FiniteMarkovRewardProcess[Position] = puzzle.apply_finite_policy(policy)
			val = implied_mrp.get_value_function_vec(gamma = 1.0)

			# store the optimal policy
			if np.linalg.norm(val, ord = 1) > max_val:
				optimal_policy = my_dict
				optimal_value = val
				max_val = np.linalg.norm(val, ord = 1)

		# plot the optimal escape-probabilities
		optimal_value = [0.] + [i for i in optimal_value] + [1.0]
		axes[j].plot(range(length + 1), optimal_value)
		axes[j].scatter(range(length + 1), optimal_value)

		ax2 = axes[j].twinx()
		policy = [0.0 if act == 'A' else 1.0 for act in optimal_policy.values()]
		ax2.scatter(range(1, length), policy, color = 'black')

		axes[j].set_title(f"Optimal escape-probability and action as a function of states for $n$ = {length}")
		axes[j].set_xlabel("State")
		axes[j].set_ylabel("Probability")
		ax2.set_ylabel("Action")
		axes[j].grid(alpha = 0.6, linestyle = ':')

	fig.tight_layout()
	plt.show()

