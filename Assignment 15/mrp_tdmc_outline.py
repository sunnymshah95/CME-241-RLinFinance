import numpy as np 
from typing import Sequence, Tuple, Mapping
from collections import defaultdict

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    counts = defaultdict(int)
    values = defaultdict(float)

    for state, return_ in state_return_samples:
        counts[state] += 1
        values[state] += (return_ - values[state]) / counts[state]

    return dict(values)



def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    total = 0
    prob_fun : ProbFunc = defaultdict(lambda : defaultdict(int))
    reward_f : RewardFunc = defaultdict(int)
    count_f  : Mapping[S, float] = defaultdict(int)

    for state, reward, next_state in srs_samples:
        prob_fun[state][next_state] += 1
        reward_f[state] += reward
        count_f[state] += 1
        total += 1

    prob_fun = {state : {next_state : outcome / total for next_state, outcome in pair.items()} for state, pair in prob_fun.items()}
    reward_f = {state : reward / count_f[state] for state, reward in reward_f.items()}

    return prob_fun, reward_f


def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    states = set(prob_func.keys())
    R = np.array([reward_func[state] for state in prob_func])
    P = np.array([[p for s, p in pairs.items() if s != 'T'] for state, pairs in prob_func.items()])
    vf : np.ndarray = np.linalg.inv(np.eye(len(states)) - P) @ R

    return {s : vf[i] for i, s in enumerate(states)}


def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    vf : ValueFunc = {s[0] : 0.0 for s in srs_samples}
    n_samples = len(srs_samples)

    for num_iter in range(num_updates):
        lr = learning_rate * (num_iter / learning_rate + 1) ** -0.5
        s, r, s1 = srs_samples[np.random.choice(range(n_samples), size=1)[0]]
        vf[s] += lr * (r + (vf[s1] if s1 != 'T' else 0.) - vf[s])

    return vf

def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    states = list(set(s[0] for s in srs_samples))
    vf : ValueFunc = {s : 0.0 for s in states}
    ffs = [(lambda x, s=s : float(x == s)) for s in states]

    m = len(states)
    A_inv : np.ndarray = np.eye(m)
    b_vec : np.ndarray = np.zeros(m)

    for state, reward, next_state in srs_samples:
        u = np.array([f(state) for f in ffs])
        v = u

        if next_state != 'T':
            v -= np.array([f(next_state) for f in ffs])

        # updating the inverse of A using Sherman-Morrison-Woodbury formula
        A_inv -= (A_inv @ np.outer(u, v) @ A_inv) / (1 + v @ A_inv @ u)
        b_vec += reward * u

    return {states[i] : v for i, v in enumerate(A_inv @ b_vec)}



if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))