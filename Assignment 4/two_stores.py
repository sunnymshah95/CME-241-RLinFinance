# from email.mime import base
import sys
sys.path.append("../")

from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hand_1: int
    on_order_1: int
    on_hand_2: int
    on_order_2: int

    def inventory_position_1(self) -> int:
        return self.on_hand_1 + self.on_order_1

    def inventory_position_2(self) -> int:
        return self.on_hand_2 + self.on_order_2


InvOrderMapping = Mapping[
    InventoryState,
    Mapping[Tuple[int, int, int], Categorical[Tuple[InventoryState, float]]]
]


class TwoStoresInventoryControl(FiniteMarkovDecisionProcess[InventoryState, Tuple[int, int, int]]):

    def __init__(
        self,
        capacity_1: int,
        capacity_2: int,
        poisson_lambda_1: float,
        poisson_lambda_2: float,
        holding_cost_1: float,
        holding_cost_2: float,
        stockout_cost_1: float,
        stockout_cost_2: float,
        transport_cost_1: float,
        transport_cost_2: float
    ):
        self.capacity_1: int = capacity_1
        self.poisson_lambda_1: float = poisson_lambda_1
        self.holding_cost_1: float = holding_cost_1
        self.stockout_cost_1: float = stockout_cost_1

        self.capacity_2: int = capacity_2
        self.poisson_lambda_2: float = poisson_lambda_2
        self.holding_cost_2: float = holding_cost_2
        self.stockout_cost_2: float = stockout_cost_2

        self.transport_cost_1: float = transport_cost_1
        self.transport_cost_2: float = transport_cost_2

        self.demand_distribution_1 = poisson(poisson_lambda_1)
        self.demand_distribution_2 = poisson(poisson_lambda_2)
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[Tuple[int, int, int], Categorical[Tuple[InventoryState, float]]]] = {}

        for alpha_1 in range(self.capacity_1 + 1):
            for beta_1 in range(self.capacity_1 + 1 - alpha_1):
                for alpha_2 in range(self.capacity_2 + 1):
                    for beta_2 in range(self.capacity_2 + 1 - alpha_2):
                        state: InventoryState = InventoryState(on_hand_1=alpha_1, on_order_1=beta_1, on_hand_2=alpha_2, on_order_2=beta_2)
                        ip_1: int = state.inventory_position_1()
                        ip_2: int = state.inventory_position_2()
                        base_reward: float = - self.holding_cost_1 * alpha_1 - self.holding_cost_2 * alpha_2
                        d1: Dict[Tuple[int, int, int], Categorical[Tuple[InventoryState, float]]] = {}

                        for order_1 in range(self.capacity_1 - ip_1 + 1):
                            for order_2 in range(self.capacity_2 - ip_2 + 1):
                                for transfer in range(-max(alpha_2, self.capacity_1 - alpha_1 - beta_1), min(alpha_1, self.capacity_2 - alpha_2 - beta_2) + 1):
                                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] = {}
                                    prob_1: float = 1 - self.demand_distribution_1.cdf(ip_1 - 1)
                                    prob_2: float = 1 - self.demand_distribution_2.cdf(ip_2 - 1)

                                    K_1 = self.transport_cost_1 if order_1 == 0 or order_2 == 0 else 0.0
                                    K_1 = min(1, order_1) * self.transport_cost_1 + min(1, order_2) * self.transport_cost_1
                                    K_2 = self.transport_cost_2 if transfer != 0 else 0.0

                                    for i in range(ip_1):
                                        for j in range(ip_2):
                                            if i < alpha_1 + beta_1 - transfer and j < alpha_2 + beta_2 + transfer:
                                                reward: float = base_reward - K_1 - K_2
                                                sr_probs_dict[(InventoryState(ip_1 - i, order_1 - transfer, ip_2 - j, order_2 + transfer), reward)] = self.demand_distribution_1.pmf(i) * self.demand_distribution_2.pmf(j)

                                            # when the inventory position for both stores are less than the ordered and transferred units
                                            elif j < alpha_2 + beta_2 + transfer:
                                                reward: float = base_reward - self.stockout_cost_1 * (prob_1 * (self.poisson_lambda_1 - ip_1) + ip_1 * self.demand_distribution_1.pmf(ip_1)) - K_1 - K_2
                                                sr_probs_dict[(InventoryState(0, order_1 - transfer, ip_2 - j, order_2 + transfer), reward)] = prob_1 * self.demand_distribution_2.pmf(j)
                                            
                                            elif i < alpha_1 + beta_1 - transfer:
                                                reward: float = base_reward - self.stockout_cost_2 * (prob_2 * (self.poisson_lambda_2 - ip_2) + ip_2 * self.demand_distribution_2.pmf(ip_2)) - K_1 - K_2
                                                sr_probs_dict[(InventoryState(ip_1 - i, order_1 - transfer, 0, order_2 + transfer), reward)] = prob_2 * self.demand_distribution_1.pmf(i)

                                    reward: float = base_reward - self.stockout_cost_1 * (prob_1 * (self.poisson_lambda_1 - ip_1) + ip_1 * self.demand_distribution_1.pmf(ip_1)) - self.stockout_cost_2 * (prob_2 * (self.poisson_lambda_2 - ip_2) + ip_2 * self.demand_distribution_2.pmf(ip_2)) - K_1 - K_2
                                    sr_probs_dict[(InventoryState(0, order_1 - transfer, 0, order_2 + transfer), reward)] = prob_1 * prob_2

                                    d1[(order_1, order_2, transfer)] = Categorical(sr_probs_dict)

                        d[state] = d1
        return d


# if __name__ == '__main__':
    # capacity_1 = 2
    # capacity_2 = 2
    # poisson_lambda_1 = 2.0
    # poisson_lambda_2 = 1.0
    # holding_cost_1 = 1.0
    # holding_cost_2 = 3.0
    # stockout_cost_1 = 10.0
    # stockout_cost_2 = 24.0
    # transport_cost_1 = 10.0
    # transport_cost_2 = 9.0

    # user_gamma = 0.9

    # si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
    #     TwoStoresInventoryControl(
    #         capacity_1=capacity_1,
    #         capacity_2=capacity_2,
    #         poisson_lambda_1=poisson_lambda_1,
    #         poisson_lambda_2=poisson_lambda_2,
    #         holding_cost_1=holding_cost_1,
    #         holding_cost_2=holding_cost_2,
    #         stockout_cost_1=stockout_cost_1,
    #         stockout_cost_2=stockout_cost_2,
    #         transport_cost_1=transport_cost_1,
    #         transport_cost_2=transport_cost_2
    #     )

    # print("MDP Transition Map")
    # print("------------------")
    # print(si_mdp)

    # fdp: FiniteDeterministicPolicy[InventoryState, Tuple[int, int, int]] = \
    #     FiniteDeterministicPolicy(
    #         {InventoryState(alpha_1, beta_1, alpha_2, beta_2): (capacity_1 - (alpha_1 + beta_1), capacity_2 - (alpha_2 + beta_2), transfer)
    #          for alpha_1 in range(capacity_1 + 1)
    #          for beta_1 in range(capacity_1 + 1 - alpha_1)
    #          for alpha_2 in range(capacity_2 + 1) 
    #          for beta_2 in range(capacity_2 + 1 - alpha_2)
    #          for transfer in range(-max(alpha_2, capacity_1 - alpha_1 - beta_1), min(alpha_1, capacity_2 - alpha_2 - beta_2) + 1)
    #         }
    # )

    # print("Deterministic Policy Map")
    # print("------------------------")
    # print(fdp)

    # implied_mrp: FiniteMarkovRewardProcess[InventoryState] =\
    #     si_mdp.apply_finite_policy(fdp)
    # print("Implied MP Transition Map")
    # print("--------------")
    # print(FiniteMarkovProcess(
    #     {s.state: Categorical({s1.state: p for s1, p in v.table().items()})
    #      for s, v in implied_mrp.transition_map.items()}
    # ))

    # print("Implied MRP Transition Reward Map")
    # print("---------------------")
    # print(implied_mrp)

    # print("Implied MP Stationary Distribution")
    # print("-----------------------")
    # implied_mrp.display_stationary_distribution()
    # print()

    # print("Implied MRP Reward Function")
    # print("---------------")
    # implied_mrp.display_reward_function()
    # print()

    # print("Implied MRP Value Function")
    # print("--------------")
    # implied_mrp.display_value_function(gamma=user_gamma)
    # print()

    # from rl.dynamic_programming import evaluate_mrp_result
    # from rl.dynamic_programming import policy_iteration_result
    # from rl.dynamic_programming import value_iteration_result

    # print("Implied MRP Policy Evaluation Value Function")
    # print("--------------")
    # print(evaluate_mrp_result(implied_mrp, gamma=user_gamma))
    # print()

    # print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    # print("--------------")
    # opt_vf_pi, opt_policy_pi = policy_iteration_result(
    #     si_mdp,
    #     gamma=user_gamma
    # )
    # print(opt_vf_pi)
    # print(opt_policy_pi)
    # print()

    # print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    # print("--------------")
    # opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
