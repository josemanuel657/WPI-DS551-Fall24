### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
# Modified By Yanhua Li on 09/09/2022 for gym==0.25.2
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    value_function = np.zeros(nS)
    
    while (True):
        new_value_function = np.zeros(nS)
        for s in range(nS):
            for a in range(nA):
                action_probability = policy[s,a]
                for probability, nextState, reward, terminal in P[s][a]:
                    new_value_function[s] += action_probability * probability * (reward + gamma * value_function[nextState])
            
            
        delta = np.linalg.norm(new_value_function - value_function, 1)
        
        if (delta < tol): break
        
        value_function = new_value_function
        
    return value_function 


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    
	############################
	# YOUR IMPLEMENTATION HERE #
    new_policy = np.zeros([nS, nA])
    
    for s in range(nS):
        Q = np.zeros(nA)
        for a in range(nA):
            for probability, nextState, reward, terminal in P[s][a]:
                Q[a] += probability * (reward + gamma * value_from_policy[nextState])

        best_action = np.argmax(Q)
        new_policy[s] = np.eye(nA)[best_action]
    #                          #
	############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
	############################
	# YOUR IMPLEMENTATION HERE #
 
    new_policy = policy.copy()
    
    stable_policy = False
    while not stable_policy:
        V = policy_evaluation(P, nS, nA, new_policy, gamma, tol)  
        improved_policy = policy_improvement(P, nS, nA, V, gamma) 
        
        # Check if the policy is stable
        if np.array_equal(new_policy, improved_policy):
            stable_policy = True
        else:
            new_policy = improved_policy.copy()

    return new_policy, V

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    V_new = V.copy()
    policy_new = np.zeros([nS, nA])
    
    while True:
        delta = 0  # Change in value function in the current iteration
        for s in range(nS):
            Q = np.zeros(nA)
            # Compute Q value for each action
            for a in range(nA):
                for probability, nextState, reward, done in P[s][a]:
                    Q[a] += probability * (reward + gamma * V_new[nextState])
            
            # Find the maximum Q value and update V_new
            best_value = np.max(Q)
            delta = max(delta, np.abs(V_new[s] - best_value))
            V_new[s] = best_value
        
        # Check for convergence
        if delta < tol:
            break
    
    # Derive the optimal policy from the final value function
    for s in range(nS):
        Q = np.zeros(nA)
        for a in range(nA):
            for probability, nextState, reward, done in P[s][a]:
                Q[a] += probability * (reward + gamma * V_new[nextState])
        
        # Get the best action(s) for each state and encode as one-hot
        best_action = np.argmax(Q)
        policy_new[s] = np.eye(nA)[best_action]  # One-hot encode the best action

    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob, _ = env.reset() # initialize the episode
        done = False
        while not done or truncated: # using "not truncated" as well, when using time_limited wrapper.
            if render:
                env.render() # render the game
                
            action = np.argmax(policy[ob])
                
            ob, reward, done, truncated, _ = env.step(action)
                
            total_rewards += reward
            
            
    return total_rewards



