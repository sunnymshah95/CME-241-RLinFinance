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
				dict_A[(Position(position = state.position + 1), 1.0)] = 1 - state.position / self.length

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
	length = 9
	puzzle : FiniteMarkovDecisionProcess[Position, str] = FrogEscape(length = length)

	# print("MDP Transition Map")
	# print("------------------")
	# print(puzzle)


	# generate all possible combinations of actions for a given position
	outcomes = [[(pos, act) for act in ['A', 'B']] for pos in range(1, length)]
	optimal_policy = None
	optimal_value = None
	max_val = - np.inf

	# then form the cartesian product of all those lists
	for i in itertools.product(*outcomes):
		# print({pos : act for pos, act in i}, end = ' ---> Val Func = ')
		my_dict : Dict[Position, str] = {Position(position = pos) : act for pos, act in i}
		policy : FiniteDeterministicPolicy[Position, str] = FiniteDeterministicPolicy(my_dict)
		implied_mrp: FiniteMarkovRewardProcess[Position] = puzzle.apply_finite_policy(policy)
		val = implied_mrp.get_value_function_vec(gamma = 1.0)
		# print(val)

		# store the optimal policy
		if np.linalg.norm(val, ord = 1) > max_val:
			optimal_policy = my_dict
			optimal_value = val
			max_val = np.linalg.norm(val, ord = 1)





	print(f"The optimal policy is:\n{optimal_policy}")