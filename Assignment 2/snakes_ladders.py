import sys
sys.path.append("../")
from typing import Mapping, Dict, Sequence, Iterable
from rl.distribution import Categorical, FiniteDistribution, Constant, Distribution
from rl.markov_process import FiniteMarkovProcess, NonTerminal, Terminal 


class SnakesAndLaddersMP(FiniteMarkovProcess[int]):

	def __init__(self, from_to : Mapping[int, int]):
		self.from_to = from_to

		super().__init__(self.get_transition_map())


	def get_transition_map(self) -> Mapping[int, FiniteDistribution[int]]:
		d : Dict[int, FiniteDistribution[int]] = {}

		for i in range(1, 100):
			state_probs_map = {}
			next_state = 0
			
			for j in range(i + 1, min(101, i + 7)):
				if j in self.from_to.keys():
					next_state = self.from_to[j]
				else:
					next_state = j

				state_probs_map[next_state] = 1. / 6.

			if i > 94:
				state_probs_map[i] = (i - 94.) / 6.

			d[i] = Categorical(state_probs_map)

		return d

	# def traces(self, start_distribution) -> Iterable[Iterable[int]]:
	# 	print('hi')



if __name__ == '__main__':
	changes_from = [1, 4, 9, 28, 36, 21, 51, 71, 80, \
				    16, 47, 49, 56, 64, 87, 93, 95, 98]
	changes_to = [38, 14, 31, 84, 44, 42, 67, 91, 100, \
				   6, 26, 11, 53, 60, 24, 73, 75, 78]

	from_to = {fr : to for fr, to in zip(changes_from, changes_to)}
	# from_to = {fr : fr for fr in changes_from}
	game = SnakesAndLaddersMP(from_to = from_to)

	print("Generated the Markov Process")
	# print("Transition Map")
	# print("----------------")
	# print(game)


	start_distribution = Constant(NonTerminal(value = 1)

	trace_1 = game.traces(start_distribution)
	print(trace_1)

	state = game.simulate(start_distribution)
	print(state)




