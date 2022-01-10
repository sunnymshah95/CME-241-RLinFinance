import sys
sys.path.append("../")
from dataclasses import dataclass
from typing import Mapping, Dict, Sequence, Iterable
from rl.distribution import Categorical, FiniteDistribution, Constant, Distribution
from rl.markov_process import FiniteMarkovProcess, NonTerminal, Terminal 


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


if __name__ == '__main__':
	L = 3
	puzzle = FrogPuzzle(L)
	print("Transition Map")
	print("----------------")
	print(puzzle)

	start_distribution = Constant(value = NonTerminal(State(position = 1)))