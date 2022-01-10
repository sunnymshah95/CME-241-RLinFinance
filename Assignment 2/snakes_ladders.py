import sys
sys.path.append("../")
from dataclasses import dataclass
from typing import Mapping, Dict, Sequence, Iterable
from rl.distribution import Categorical, FiniteDistribution, Constant, Distribution
from rl.markov_process import FiniteMarkovProcess, NonTerminal, Terminal 


@dataclass(frozen=True)
class State:
	position : int 



class SnakesAndLaddersMP(FiniteMarkovProcess[State]):
	def __init__(self, from_to : Mapping[State, State]):
		self.from_to = from_to

		super().__init__(self.get_transition_map())


	def get_transition_map(self) -> Mapping[State, FiniteDistribution[State]]:
		d : Dict[State, FiniteDistribution[State]] = {}

		for state in range(1, 100):
			state_probs_map = {}
			next_state = 0
			
			for j in range(state + 1, min(101, state + 7)):
				if j in self.from_to.keys():
					next_state = self.from_to[j]
				else:
					next_state = j

				state_probs_map[State(position = next_state)] = 1. / 6.

			if state > 94:
				state_probs_map[State(position = state)] = (state - 94.) / 6.

			d[State(position = state)] = Categorical(state_probs_map)

		return d

	# def simulate(self, start_state_distribution):
	# 	state = start_state_distribution.sample()
	# 	print(state)
	# 	yield state

	# 	while isinstance(state, NonTerminal):
	# 		state = self.transition(state).sample()
	# 		print(state)
	# 		yield state

	# 		if isinstance(state, Terminal):
	# 			break

	def traces(self, start_state_distribution, count = 2):
		num = 1
		while True:
			yield self.simulate(start_state_distribution)
			num += 1

			if num > count:
				break



if __name__ == '__main__':
	changes_from = [1, 4, 9, 28, 36, 21, 51, 71, 80, \
				    16, 47, 49, 56, 64, 87, 93, 95, 98]
	changes_to = [38, 14, 31, 84, 44, 42, 67, 91, 100, \
				   6, 26, 11, 53, 60, 24, 73, 75, 78]

	from_to = {fr : to for fr, to in zip(changes_from, changes_to)}
	game = SnakesAndLaddersMP(from_to = from_to)

	print("Generated the Markov Process")
	# print("Transition Map")
	# print("----------------")
	# print(game)


	start_distribution = Constant(value = NonTerminal(State(position = 1)))

	outcomes = [[i for i in trace] for trace in game.traces(start_distribution, 10)]
	for i, out in enumerate(outcomes):
		print(f"Trace {i + 1}: Length of the game was {len(out)}")


