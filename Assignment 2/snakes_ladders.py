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

		for i in range(1, 100):
			state_probs_map = {}
			next_state = 0
			
			for j in range(i + 1, min(101, i + 7)):
				if j in self.from_to.keys():
					next_state = self.from_to[j]
				else:
					next_state = j

				state_probs_map[State(position = next_state)] = 1. / 6.

			if i > 94:
				state_probs_map[State(position = i)] = (i - 94.) / 6.

			d[State(position = i)] = Categorical(state_probs_map)

		return d

	def simulate(self, start_state_distribution):
		state = start_state_distribution.sample()
		print('hi')
		print(state)
		yield state

		while isinstance(state, NonTerminal):
			print('hi')
			state = self.transition(state).sample()
			yield state



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


	start_distribution = Constant(value = NonTerminal(State(position = 1)))

	tracer = game.traces(start_distribution)
	print(f"tracer = {tracer}")
	print(f"Next(tracer) = {next(tracer)}")
	print(f"Next(next(tracer)) = {next(next(tracer))}")
	print(f"Next(next(tracer)) = {next(next(tracer))}")
	print(f"Next(next(tracer)) = {next(next(tracer))}")
	mp = game.transition(state = NonTerminal(State(position = 1)))
	# print(mp)
	print(mp.sample())




