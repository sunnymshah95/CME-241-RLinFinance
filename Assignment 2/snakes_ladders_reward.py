import sys
sys.path.append("../")
from dataclasses import dataclass
from typing import Mapping, Dict, Tuple
from rl.distribution import Categorical, FiniteDistribution, Constant, Distribution
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess, NonTerminal
import matplotlib.pyplot as plt
import itertools


@dataclass(frozen=True)
class State:
	position : int 



class SnakesAndLaddersMRP(FiniteMarkovRewardProcess[State]):
	def __init__(self, from_to : Mapping[State, State]):
		self.from_to = from_to
		super().__init__(self.get_transition_reward_map())
		print("Generated the Markov Reward Process")


	def get_transition_reward_map(self) -> Mapping[State, FiniteDistribution[Tuple[State, float]]]:
		d : Dict[State, FiniteDistribution[Tuple[State, float]]] = {}
		reward = 1

		for state in range(1, 100):
			state_probs_map = {}
			next_state = 0
			
			for j in range(state + 1, min(101, state + 7)):
				if j in self.from_to.keys():
					next_state = self.from_to[j]
				else:
					next_state = j

				state_probs_map[(State(position = next_state), reward)] = 1. / 6.

			if state > 94:
				state_probs_map[(State(position = state), reward)] = (state - 94.) / 6.

			d[State(position = state)] = Categorical(state_probs_map)

		return d



if __name__ == '__main__':
	changes_from = [1, 4, 9, 28, 36, 21, 51, 71, 80, \
				    16, 47, 49, 56, 64, 87, 93, 95, 98]
	changes_to = [38, 14, 31, 84, 44, 42, 67, 91, 100, \
				   6, 26, 11, 53, 60, 24, 73, 75, 78]

	from_to = {fr : to for fr, to in zip(changes_from, changes_to)}
	game = SnakesAndLaddersMRP(from_to = from_to)
	gamma = 1.0 # the discount factor

	# print("Transition Map")
	# print("--------------")
	# print(FiniteMarkovProcess({s.state: Categorical({s1.state: p for s1, p in v.table().items()}) for s, v in game.transition_map.items()}))

	
	# print("Transition Reward Map")
	# print("----------------")
	# print(game)


	# print("Reward Function")
	# print("----------------")
	# game.display_reward_function()

	# print("Value Function")
	# print("----------------")
	# game.display_value_function(gamma = gamma)

	val = game.get_value_function_vec(gamma = gamma)
	print(f"The expected number of dice rolls (using the value function) is {val[0]:.3f}.")


	start_distribution = Constant(value = NonTerminal(State(position = 1)))
	num_traces = 10000
	outcomes = [len([st for st in it]) for it in itertools.islice(game.reward_traces(start_distribution), num_traces)]
	print(f"The expected number of dice rolls (using Monte Carlo)        is {sum(outcomes) / num_traces:.3f}.")