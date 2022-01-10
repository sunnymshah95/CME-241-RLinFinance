import sys
sys.path.append("../")
from dataclasses import dataclass
from typing import Mapping, Dict, Sequence, Iterable
from rl.distribution import Categorical, FiniteDistribution, Constant, Distribution
from rl.markov_process import FiniteMarkovProcess, NonTerminal, Terminal 
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class State:
	position : int


class FrogPuzzle(FiniteMarkovProcess[State]):
	length : int = 0 # this is the length of the river between the two banks

	def __init__(self, length : int):
		self.length = length
		super().__init__(self.get_transition_map())
		print("Frog puzzle has been created.")


	def get_transition_map(self) -> Mapping[State, FiniteDistribution[State]]:
		d : Dict[State, FiniteDistribution[State]] = {}

		for state in range(1, self.length + 2):
			state_prob_map = {}
			for next_state in range(state + 1, self.length + 3):
				state_prob_map[State(position=next_state)] = 1. / (self.length - state + 2)

			d[State(position = state)] = Categorical(state_prob_map)

		return d

	def traces(self, start_state_distribution, num_traces = 100000):
		num = 1
		while True:
			yield self.simulate(start_state_distribution)
			num += 1

			if num > num_traces:
				break


if __name__ == '__main__':
	L = 10
	puzzle = FrogPuzzle(L)
	# print("Transition Map")
	# print("----------------")
	# print(puzzle)

	start_distribution = Constant(value = NonTerminal(State(position = 1)))
	num_traces = 10000

	outcomes = [len([i for i in trace]) for trace in puzzle.traces(start_distribution, num_traces)]
	# print(outcomes)

	plt.hist(outcomes)
	plt.show()