from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################
    V = initV.copy()
    gamma = env_spec.gamma

    for traj in trajs:
        T = len(traj)
        for t in range(T):
            G = 0
            for i in range(t, min(t + n, T)):
                G += (gamma ** (i - t)) * traj[i][2]  # r_{i+1} is traj[i][2]
            if t + n < T:
                G += (gamma ** n) * V[traj[t + n][0]]  # s_{t+n}
            V[traj[t][0]] += alpha * (G - V[traj[t][0]])  # s_t

    return V

    

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################
    Q = initQ.copy()
    gamma = env_spec.gamma
    nS = env_spec.nS
    nA = env_spec.nA

    class GreedyPolicy(Policy):
        def action(self, state: int) -> int:
            return np.argmax(Q[state])

        def action_prob(self, state: int, action: int) -> float:
            return 1.0 if action == np.argmax(Q[state]) else 0.0

    pi_star = GreedyPolicy()

    for traj in trajs:
        T = len(traj)
        for t in range(T):
            G = 0
            rho = 1
            for i in range(t, min(t + n, T)):
                G += (gamma ** (i - t)) * traj[i][2]  # r_{i+1} is traj[i][2]
                if i + 1 < T:
                    rho *= pi_star.action_prob(traj[i][0], traj[i][1]) / bpi.action_prob(traj[i][0], traj[i][1])
            if t + n < T:
                G += (gamma ** n) * Q[traj[t + n][0], np.argmax(Q[traj[t + n][0]])]  # (s_{t+n}, a_{t+n})
            Q[traj[t][0], traj[t][1]] += alpha * rho * (G - Q[traj[t][0], traj[t][1]])  # (s_t, a_t)

    pi_star = GreedyPolicy()
    return Q, pi_star

    
