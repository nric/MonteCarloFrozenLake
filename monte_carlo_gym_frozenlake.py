#%%
""" 
This is a monte carlo controll first visit solution to discrete toy text open ai gym enviroment FrozenLake-v0. 
Theoretically, it should be able to solve all toy text enviroments but here it is tested for Frozen Lake.
It is written using visual studio code with Jupyter Kernel connection (Python interactive). 
Just run the cell. If you want to run using python interpreter directly, replace def main(): to if name == 'main': and remove the last line (call of main()).
"""
import gym
import numpy as np
from itertools import product


def epsilon_action(a,env,epsilon = 0.1):
    """ 
    Return the action most of the time but in 1-epsiolon of the cases a random action within the env.env.action_space is returned
    Return: action
    """
    rand_number = np.random.random()
    if rand_number < (1-epsilon):
        return a
    else:
        return env.action_space.sample()

def play_a_game(env,policy,epsilon=0.1):
    """ 
    Returns the lived S,A,R of an episode (as a list of a list). In monte carlo the path is epsion-greed partly random.
    Args: env: Gym enviroment policy: the current policy
    Return: List of all states with S,A,R for each.
    """
    env.reset()
    finished = False
    episode_sar = []
    while not finished:
        current_s= env.env.s
        action = epsilon_action(policy[current_s],env,epsilon=epsilon)
        new_s, reward, finished, _info =  env.step(action)
        episode_sar.append([current_s,action,reward])
    #episode_sar.append([new_s,None,reward])
    return episode_sar

def sar_to_sag(sar_list,GAMMA=0.9):
    """ 
    The gain G in Monte Carlo is caluclates by means of the reward for each state and a discount factor Gamma for earlier episondes.
    Careful: SAR list needs to be reversed for correct calculation of G
    Args: sar_list: List of S,A,R values of this episode. Gamma: discount factor for future episodes
    Return: List of all states with S,A,G for each state visited.
    """
    G = 0
    state_action_gain = []
    for s,a,r in reversed(sar_list):
        G = r + GAMMA*G
        state_action_gain.append([s,a,G])
    return reversed(state_action_gain)


def monte_carlo(env, episodes=10000, epsilon=0.1):
    """ 
    Function for generating a policy the monte carlo way: Play a lot, find the optimal policy this way
    Args: env: the open ai gym enviroment object
    Return: policy: the "optimal" policy V: the value table for each s (optional)
    """
    #create a random policy
    policy = {j:np.random.choice(env.action_space.n) for j in range(env.observation_space.n)} 
    #Gain or return is cummulative rewards over the entiere episode g(t) = r(t+1) + gamma*G(t+1)
    G = 0
    #Q function is essential for the policy update
    Q = {j:{i:0 for i in range(env.action_space.n)} for j in range(env.observation_space.n)} 
    #The s,a pairs of the Q function are updated using mean of returns of each episode. So returns need to be collected
    returns = {(s,a):[] for s,a in product(range(env.observation_space.n),range(env.action_space.n))}

    for ii in range(episodes):
        seen_state_action_pairs = set()
        #play a game and convert S,A,R to S,A,G
        episode_sag = sar_to_sag(play_a_game(env,policy,epsilon=epsilon if ii > 1000 else 1))        
        #Use S,A,G to update Q (first-visit method), retruns and seen_state_action_paris
        for s,a,G in episode_sag:
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                seen_state_action_pairs.add(sa)
        # calculate new policy p[s] = argmax[a]{Q(s,a)}
        for s in policy.keys():
            policy[s] = max(Q[s],key=Q[s].get)

    #optional: create V[s]
    V = {s:max(list(Q[s].values())) for s in policy.keys()}

    return policy, V

def test_policy(env,policy):
    env.reset()
    finished = False
    while not finished:
        _new_s, _reward, finished, _info =  env.step(policy[env.env.s])
        env.render()
        if finished: break

def main():
    #env = gym.make('FrozenLake8x8-v0')
    env = gym.make('FrozenLake-v0')
    env.render()
    policy, V = monte_carlo(env,episodes=10000,epsilon=0.1)   
    print(policy)
    test_policy(env,policy)

main()