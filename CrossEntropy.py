import numpy as np
import gym

def cross_entropy_method(env, n_iterations=100, n_samples=50, elite_frac=0.2, gamma=0.99):
    """
    Cross-entropy method for policy optimization in a gym environment.

    :param env: Gym environment
    :param n_iterations: Number of iterations for the optimization
    :param n_samples: Number of samples to generate in each iteration
    :param elite_frac: Fraction of samples to consider as elite
    :param gamma: Discount factor for future rewards
    :return: Optimized policy parameters
    """
    n_elite = int(n_samples * elite_frac)
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    policy_params = np.random.rand(n_states, n_actions)  # Randomly initialize policy parameters

    for iteration in range(n_iterations):
        rewards = np.zeros(n_samples)
        episodes = []

        # Generating episodes
        for i in range(n_samples):
            episode_rewards = 0
            state = env.reset()
            done = False
            episode = []

            while not done:
                probabilities = np.dot(state, policy_params)
                action = np.argmax(probabilities)
                state, reward, done, _ = env.step(action)
                episode_rewards += reward * (gamma ** len(episode))
                episode.append((state, action, reward))

            episodes.append(episode)
            rewards[i] = episode_rewards

        # Selecting elite episodes
        elite_indices = rewards.argsort()[-n_elite:]
        elite_episodes = [episodes[i] for i in elite_indices]

        # Update policy parameters
        new_params = np.zeros_like(policy_params)
        for episode in elite_episodes:
            for state, action, _ in episode:
                new_params[:, action] += state
        policy_params = new_params / n_elite

    return policy_params

# Example usage
env_name = "CartPole-v1"
env = gym.make(env_name)
optimized_params = cross_entropy_method(env)
print("Optimized Policy Parameters:", optimized_params)
