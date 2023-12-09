import pickle
import sys
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
from env.cartpole import CartPole
from env.pendulum import PendulumEnv
from env.gridworld import GW687
from env.boxpushing import roboBoxPushing
import env.gridworld as gw


from torch import nn, optim, relu, from_numpy, distributions, tensor
class SoftmaxPolicyNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(SoftmaxPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1) # dim=1 fixed issues; TODO: verify its not dim=-1

    def forward(self, state):
        state = relu(self.fc1(state))
        return self.softmax(self.fc2(state))
    
class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        state = relu(self.fc1(state))
        return self.fc2(state)

# def AC_one_step(env, alpha_theta:float = .01, alpha_w:float = .01, n_episodes=100, gamma=0.99, max_time_step:int=500):
#     """
#     One step Actor-Critic

#     :param env: Gym environment
#     :param n_iterations: Number of iterations for the optimization
#     :param gamma: Discount factor for future rewards
#     :return: Optimized policy parameters
#     """
#     n_actions = env.action_space.n
#     n_states = env.observation_space.shape[0]

#     policy_net = SoftmaxPolicyNetwork(state_size=n_states, hidden_size=128, action_size=n_actions)
#     value_net = ValueNetwork(state_size=n_states, hidden_size=128)
#     policy_optimizer = optim.Adam(policy_net.parameters(), lr=alpha_theta)
#     value_optimizer = optim.Adam(value_net.parameters(), lr=alpha_w)

#     all_total_rewards = []  # Store returns for each episode for logging
#     average_policy_losses = []
#     average_value_losses = []
#     for i in range(n_episodes):
#         print(i)
#         # Generate trajectory
#         state = env.reset()[0]
#         log_probs = []
#         values = []
#         rewards = []
#         policy_losses = []
#         value_losses = []
#         done = False
#         t = 0
#         while not done:
#             if isinstance(state, tuple):
#                 state = np.array(state)
#             # Turn state to Tensor
#             state_tensor = from_numpy(state).float().unsqueeze(0)
#             action_probs = policy_net(state_tensor)
#             value = value_net(state_tensor)

#             # Always assume categorical action distribution
#             dist = distributions.Categorical(action_probs)
#             action = dist.sample()
#             log_prob = dist.log_prob(action)
#             next_state, reward, done, _, _ = env.step(action.item())

#             next_state_tensor = from_numpy(next_state).float().unsqueeze(0)
#             next_value = value_net(next_state_tensor)
#             delta = reward + gamma*next_value*(1 - int(done)) - value

#             # Update the NN weights
#             policy_optimizer.zero_grad()
#             value_optimizer.zero_grad()

#             # Ignore gamma term since not needed in practice
#             policy_loss = -log_prob * delta.detach() # Detach delta to prevent it from influencing the policy gradient
#             policy_losses.append(policy_loss)

#             # Use mean-squared error
#             value_loss = delta.pow(2).mean()
#             value_losses.append(value_loss)

#             # Perform backpropagation
#             policy_loss.backward()
#             value_loss.backward()
#             policy_optimizer.step()
#             value_optimizer.step()

#             # Append to lists for logging
#             log_probs.append(log_prob)
#             values.append(value)
#             rewards.append(reward)
#             state = next_state

#             t += 1
#             # Set max time step
#             if t == max_time_step:
#                 break

#         # sum_policy_loss = sum(policy_losses)
#         # sum_value_loss = sum(value_losses)

#         # policy_optimizer.zero_grad()
#         # baseline_optimizer.zero_grad()

#         # # Perform a single backward pass
#         # policy_loss_accumulated.backward()
#         # value_loss_accumulated.backward()

#         # # Take an optimization step
#         # policy_optimizer.step()
#         # baseline_optimizer.step()

#         total_reward = sum(rewards)
#         all_total_rewards.append(total_reward)

#         average_policy_loss = sum(policy_losses)/len(policy_losses)
#         average_value_loss = sum(value_losses)/len(value_losses)

#         average_policy_losses.append(average_policy_loss.item())
#         average_value_losses.append(average_value_loss.item())

#     return policy_net, value_net, all_total_rewards, average_policy_losses, average_value_losses

def AC_one_step(env, alpha_theta:float = .01, alpha_w:float = .01, n_episodes=100, gamma=0.99, max_time_step:int=500):
    """
    One step Actor-Critic

    :param env: Gym environment
    :param n_iterations: Number of iterations for the optimization
    :param gamma: Discount factor for future rewards
    :return: Optimized policy parameters
    """
    try:
        # For gym envs
        n_actions = env.action_space.n
        n_states = env.observation_space.shape[0]
    except:
        # For our custom envs
        n_actions = env.nA
        n_states = env.nS


    policy_net = SoftmaxPolicyNetwork(state_size=n_states, hidden_size=128, action_size=n_actions)
    value_net = ValueNetwork(state_size=n_states, hidden_size=128)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=alpha_theta)
    value_optimizer = optim.Adam(value_net.parameters(), lr=alpha_w)

    all_total_rewards = []  # Store returns for each episode for logging
    average_policy_losses = []
    average_value_losses = []
    for i in range(n_episodes):
        # Generate trajectory
        # state = env.reset()[0] # This works for real gym envs
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        policy_losses = []
        value_losses = []
        done = False
        t = 0

        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()

        while not done:
            if isinstance(state, tuple):
                state = np.array(state)
            # Turn state to Tensor
            state_tensor = from_numpy(state).float().unsqueeze(0)
            action_probs = policy_net(state_tensor)
            value = value_net(state_tensor)

            # Always assume categorical action distribution
            dist = distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            # next_state, reward, done, _, _ = env.step(action.item()) # Note: Works for actual gym envs
            next_state, reward, done = env.step(action.item()) 

            next_state_tensor = from_numpy(next_state).float().unsqueeze(0)
            next_value = value_net(next_state_tensor)
            target = reward + gamma*next_value.item()*(1 - int(done))
            delta = target - value.item()

            # Ignore gamma term since not needed in practice
            policy_loss = -log_prob * delta # Detach delta to prevent it from influencing the policy gradient
            policy_losses.append(policy_loss)

            # Use mean-squared error
            value_loss = (reward + gamma*next_value.squeeze()*(1 - int(done)) - value.squeeze()).pow(2).mean()
            value_losses.append(value_loss)

            # Append to lists for logging
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state

            t += 1
            # Set max time step
            if t == max_time_step:
                break

        print(str(i) + "-" + str(t))
        # Sum up the losses
        total_policy_loss = torch.stack(policy_losses).sum()
        total_value_loss = torch.stack(value_losses).sum()

        total_policy_loss.backward()
        total_value_loss.backward()

        policy_optimizer.step()
        value_optimizer.step()

        total_reward = sum(rewards)
        all_total_rewards.append(total_reward)

        average_policy_loss = sum(policy_losses)/len(policy_losses)
        average_value_loss = sum(value_losses)/len(value_losses)

        average_policy_losses.append(average_policy_loss.item())
        average_value_losses.append(average_value_loss.item())

    return policy_net, value_net, all_total_rewards, average_policy_losses, average_value_losses


class Parameters():
    def __init__(self, alpha_theta, alpha_w, n_episodes, gamma):
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.n_episodes = n_episodes
        self.gamma = gamma

def random_tune(env, iterations: int, n_episodes: int, gamma: float, max_time_steps: int):
    J_values_dict = {}
    best_avg_return = 0
    for i in range(0, iterations):
        print(f'iteration: {i}')
        # Hyperparamter tune via Guassian values
        alpha_theta = np.random.uniform(.0001, .01)
        alpha_w = np.random.uniform(.0001, .01)
        _, _, J_values, _, _ = AC_one_step(env, alpha_theta=alpha_theta, alpha_w=alpha_w, n_episodes=n_episodes, gamma=gamma, max_time_step=max_time_steps)

        average_return = sum(J_values) / len(J_values)
        if average_return > best_avg_return:
            best_avg_return = average_return

        print(f"average return: {average_return}; best return: {best_avg_return}")
        J_values_dict[Parameters(alpha_theta, alpha_w, n_episodes, gamma)] = J_values

    return J_values_dict

def plot_line_graph(env_name: str, data_vector: list, std_dev: list, n_iterations: int, n_episodes: int, misc: str, name: str):
    """
    Plots a line graph given a data vector with std deviation.
    """
    # Convert lists to numpy arrays for element-wise operations
    data_vector = np.array(data_vector)
    std_dev = np.array(std_dev)

    x = np.arange(len(data_vector))
    plt.plot(x, data_vector)
    # Filling between the mean + std_dev and mean - std_dev
    plt.fill_between(x, data_vector - std_dev, data_vector + std_dev, color='blue', alpha=0.2, label='Standard Deviation')
    plt.xlabel('Episode')
    plt.ylabel(misc)
    plt.title(f'{misc} over {n_iterations} iterations with {n_episodes} episodes')
    plt.grid(True)
    # plt.savefig(f"AC/figs/{env_name}/plot-{name}.png", dpi=300)  # Saves as a PNG file with high resolution
    plt.show()

def plot(env, env_name, n_iterations: int, alpha_theta: float, alpha_w: float, n_episodes: int, gamma: float, max_time_steps: int):
    """Plots n iterations with std deviation given algorithm parameters"""
    reward_list_values = []
    policy_loss_values = []
    value_loss_values = []
    for _ in range(0, n_iterations):
        _, _, all_total_rewards, average_policy_losses, average_value_losses = AC_one_step(env, alpha_theta=alpha_theta, alpha_w=alpha_w, n_episodes=n_episodes, gamma=gamma, max_time_step=max_time_steps)
        reward_list_values.append(all_total_rewards)
        policy_loss_values.append(average_policy_losses)
        value_loss_values.append(average_value_losses)

    all_lists=[reward_list_values, policy_loss_values, value_loss_values]
    list_names = ["reward", "policy_loss", "value_loss"]
    for list_name, list_values in zip(list_names, all_lists):
        values = average_lists(list_values)
        std_values = std_dev_lists(list_values)
        # J_values = max_lists(J_values)
        plot_line_graph(env_name, values, std_values, n_iterations=n_iterations, n_episodes=n_episodes, misc=f"Average total {list_name}", name=list_name)

def std_dev_lists(list_of_lists):
    """Returns a list where each element is the standard deviation of the elements at the corresponding indices in the input lists."""
    return [np.std(values) for values in zip(*list_of_lists)]

def average_lists(list_of_lists):
    """Returns a list where each element is the average of the elements at the corresponding indices in the input lists."""
    return [sum(values)/len(values) for values in zip(*list_of_lists)]

def max_lists(list_of_lists):
    """Returns a list where each element is the max of the elements at the corresponding indices in the input lists."""
    return [max(values) for values in zip(*list_of_lists)]

if __name__=="__main__":
    # Example usage
    env_name = "CartPole"
    env = CartPole()
    n_episodes = 2000
    gamma = .99
    max_time_steps = 500

    DUMP = True
    filename = f'AC/data/{env_name}/data-{n_episodes}.pkl'
    if DUMP:
        # DUMPING
        # Load the existing dictionary from the pickle file (or start with an empty dictionary if the file doesn't exist)
        try:
            with open(filename, 'rb') as fp:
                data = pickle.load(fp)
        except (FileNotFoundError, EOFError):
            data = {}

        # Your new data
        values_dict = random_tune(env, iterations=5, n_episodes=n_episodes, gamma=gamma, max_time_steps=max_time_steps)

        # Update the dictionary with the new data
        data.update(values_dict)

        # Save the updated dictionary back to the pickle file
        with open(filename, 'wb') as fp:
            pickle.dump(data, fp)

    # READING
    # Read the pickled dictionary
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)

    # Calculate the average for each list and sort by it
    sorted_data = dict(sorted(loaded_data.items(), key=lambda item: sum(item[1])/len(item[1]), reverse=True))
    top_items = list(sorted_data.items())[0:1]
    for key, value in top_items:
        print(f"Key: {key.alpha_theta, key.alpha_w, key.n_episodes, key.gamma}, Value: {value}")
        plot(env, env_name, n_iterations=5, alpha_theta=key.alpha_theta, alpha_w=key.alpha_w, n_episodes=key.n_episodes, gamma=key.gamma, max_time_steps=max_time_steps)

