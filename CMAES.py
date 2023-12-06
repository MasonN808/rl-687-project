import numpy as np
import gym

def pi2(env, n_iterations=100, n_rollouts=50, sigma=0.1, gamma=0.99, lambda_param=1.0):
    """
    PI^2 method for policy optimization in a gym environment.

    :param env: Gym environment
    :param n_iterations: Number of iterations for the optimization
    :param n_rollouts: Number of rollouts per iteration
    :param sigma: Standard deviation for action exploration
    :param gamma: Discount factor for future rewards
    :param lambda_param: Lambda parameter for the PI^2 algorithm
    :return: Optimized policy parameters
    """
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    policy_params = np.random.rand(n_states, n_actions)  # Initialize policy parameters
    sigma_matrix = sigma * np.eye(n_states)  # Initialize covariance matrix

    for iteration in range(n_iterations):
        rollouts = []
        costs = np.zeros(n_rollouts)

        # Exploration Phase
        for k in range(n_rollouts):
            state = env.reset()
            done = False
            rollout = []
            cost = 0

            while not done:
                action_prob = np.dot(state, policy_params)
                action = np.random.choice(n_actions, p=action_prob)
                next_state, reward, done, _ = env.step(action)
                rollout.append((state, action, reward))
                cost += reward * (gamma ** len(rollout))
                state = next_state

            rollouts.append(rollout)
            costs[k] = cost

        # Parameter Update Phase
        for i in range(len(rollouts[0])):
            # Compute S and P for each rollout at time step i
            S = np.array([sum(gamma ** j * rollout[j][2] for j in range(i, len(rollout))) for rollout in rollouts])
            P = np.exp(-1 / lambda_param * S)
            P /= np.sum(P)

            # Update policy parameters
            theta_updates = np.sum(P[:, None, None] * np.array([rollout[i][0] for rollout in rollouts]), axis=0)
            policy_params += theta_updates

            # Update covariance matrix
            theta_diff = np.array([rollout[i][0] - policy_params for rollout in rollouts])
            sigma_matrix += np.sum(P[:, None, None] * theta_diff[:, :, None] * theta_diff[:, None, :], axis=0)

        # Normalize the updates over time
        n_time_steps = len(rollouts[0])
        for i in range(n_time_steps):
            policy_params *= (n_time_steps - i) / n_time_steps
            sigma_matrix *= (n_time_steps - i) / n_time_steps

    return policy_params, sigma_matrix

# Example usage
env_name = "CartPole-v1"
env = gym.make(env_name)
optimized_params, optimized_sigma = pi2(env)
print("Optimized Policy Parameters:", optimized_params)
print("Optimized Sigma Matrix:", optimized_sigma)
