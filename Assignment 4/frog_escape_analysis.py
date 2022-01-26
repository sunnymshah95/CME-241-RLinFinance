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
from rl.dynamic_programming import policy_iteration, value_iteration
from time import time 

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

def using_value_iteration(length : int, TOLERANCE : float) -> float:
	'''
	this function times the value_iteration algorithm for a given length

	parameters:
	-----------
	@length: the length of the river for the MDP
	@TOLERANCE: the tolerance for stopping the iteration

	returns:
	--------
	@duration: the time it took to run the algorithm (in milliseconds)
	'''
	start_time = time()
	frog_mdp : FiniteMarkovDecisionProcess[Position, str] = FrogEscape(length = length)
	old_vf : Dict[Position, float] = {s: 0.0 for s in frog_mdp.non_terminal_states}
	vf_generator = value_iteration(mdp = frog_mdp, gamma = 1.0)
	new_vf = next(vf_generator)
	for new_vf in vf_generator:
		if done(old_vf, new_vf, TOLERANCE):
			break
		old_vf = new_vf

	return (time() - start_time) * 1000.0


def using_policy_iteration(length : int, TOLERANCE : float) -> float:
	'''
	this function times the policy_iteration algorithm for a given length

	parameters:
	-----------
	@length: the length of the river for the MDP
	@TOLERANCE: the tolerance for stopping the iteration

	returns:
	--------
	@duration: the time it took to run the algorithm (in milliseconds)
	'''
	start_time = time()
	frog_mdp : FiniteMarkovDecisionProcess[Position, str] = FrogEscape(length = length)
	old_vf = {s: 0.0 for s in frog_mdp.non_terminal_states}
	vf_generator = policy_iteration(mdp = frog_mdp, gamma = 1.0)
	new_vf = next(vf_generator)
	for new_vf, new_pi in vf_generator:
		if done(old_vf, new_vf, TOLERANCE):
			break
		old_vf = new_vf

	return (time() - start_time) * 1000.0



def done(v1 : Dict[Position, float], v2 : Dict[Position, float], tol : float):
	'''
	this function takes in two dictionaries, converts them to numpy
	arrays and then returns True if the maximum absolute value across
	one array and the other is smaller than the specified tolerance

	parameters:
	v1: a dictionary with states as the keys and the elements
		in the value function as the values
	v2: a dictionary with states as the keys and the elements
		in the value function as the values
	tol: the specified tolerance

	returns:
	--------
	True if the maximum difference is less than TOLERANCE
	False otherwise
	'''
	array1 = np.array([i for i in v1.values()])
	array2 = np.array([i for i in v2.values()])

	return np.linalg.norm(array1 - array2, ord = np.inf) < tol

if __name__ == '__main__':
	TOLERANCE = 1e-2
	n_sim = 3 # number of simulations to run for finding average time
	lengths = range(10, 151, 10)
	value_times = [] # will store the average time for value iteration
	policy_times = [] # will store the average time for policy iteration

	# for each length, run n_sim number of simulations
	# and then calculate the average time taken for
	# the algorithm to converge
	for length in lengths:
		values_tmp = []
		policy_tmp = []

		for _ in range(n_sim):
			values_tmp.append(using_value_iteration(length, TOLERANCE))
			policy_tmp.append(using_policy_iteration(length, TOLERANCE))

		value_times.append(np.mean(values_tmp))
		policy_times.append(np.mean(policy_tmp))

	# plotting the graph of the size of state space
	# versus the time taken to run the algorithm!
	fig, ax1 = plt.subplots()
	ax1.plot(lengths, value_times, label = 'Value Iteration', color = 'blue')
	ax1.scatter(lengths, value_times, color = 'blue')

	ax2 = ax1.twinx()

	ax2.plot(lengths, policy_times, label = 'Policy Iteration', color = 'red')
	ax2.scatter(lengths, policy_times, color = 'red')

	fig.suptitle("Convergence speed versus size of problem")
	ax1.set_xlabel("Size of state space of MDP, $n$")
	ax1.set_ylabel("Time for Value Iteration  (ms)")
	ax2.set_ylabel("Time for Policy Iteration (ms)")

	ax1.legend(loc = 'upper left')
	ax2.legend(loc = 'upper right')
	plt.grid(alpha = 0.75, linestyle = ":")
	plt.show()
	# plt.savefig('complexity_graph.png')


