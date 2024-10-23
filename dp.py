from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)   
    #####################
    V = np.copy(initV)
    Q = np.zeros((env.spec.nS, env.spec.nA))
    
    while True:
        delta = 0
        for s in range(env.spec.nS):
            v = V[s]
            V[s] = sum(pi.action_prob(s, a) * sum(env.TD[s, a, s_prime] * 
                    (env.R[s, a, s_prime] + env.spec.gamma * V[s_prime]) 
                    for s_prime in range(env.spec.nS)) 
                    for a in range(env.spec.nA))
            delta = max(delta, abs(v - V[s]))
        
        # Update the Q-function
        for s in range(env.spec.nS):
            for a in range(env.spec.nA):
                Q[s, a] = sum(env.TD[s, a, s_prime] *
                              (env.R[s, a, s_prime] + env.spec.gamma * V[s_prime])
                              for s_prime in range(env.spec.nS))
        
        if delta < theta:
            break

    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################
    V = np.copy(initV)
    policy_stable = False

    while not policy_stable:
        delta = 0
        for s in range(env.spec.nS):
            v = V[s]
            # Update V[s] to the maximum action value
            V[s] = max(sum(env.TD[s, a, s_prime] *
                           (env.R[s, a, s_prime] + env.spec.gamma * V[s_prime])
                           for s_prime in range(env.spec.nS))
                       for a in range(env.spec.nA))
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            policy_stable = True

    # Derive the policy from the value function
    pi = PolicyDerivedFromValueFunction(V, env)
    
    


    return V, pi

class PolicyDerivedFromValueFunction(Policy):
    def __init__(self, V, env):
        self.V = V
        self.env = env
    
    def action_prob(self, state, action):
        best_action = self.action(state)
        return 1 if action == best_action else 0
    
    def action(self, state):
        # Choose the action that maximizes the current value function
        action_values = np.array([
            sum(self.env.TD[state, a, s_prime] *
                (self.env.R[state, a, s_prime] + self.env.spec.gamma * self.V[s_prime])
                for s_prime in range(self.env.spec.nS))
            for a in range(self.env.spec.nA)
        ])
        return np.argmax(action_values)
