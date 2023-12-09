import numpy as np
from env.cartpole import CartPole
from env.pendulum import PendulumEnv
from env.gridworld import GW687
from env.boxpushing import roboBoxPushing
from collections import defaultdict,  deque, namedtuple
import copy
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Neural network for Q-value approximation
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)
    
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    

class FAPolicy(object):
    def __init__(self, nS, nA, lr=0.0005):
        self.Q1 = DQN(nS, nA)
        self.Q2 = DQN(nS, nA)
        self.optimizer = optim.Adam(self.Q1.parameters(), lr=lr)
        self.nS = nS
        self.nA = nA
        self.update_target_model()
        self.TARGET_UPDATE = 5

    def update_target_model(self):
        self.Q2.load_state_dict(self.Q1.state_dict())

    def soft_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.nA)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float32)
                q_values = self.Q1(state_tensor)
                return q_values.max(1)[1].item()

    def greedy_action(self, state, model):
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.float32)
            q_values = model(state_tensor)
            return q_values.max(1)[1].item()

    def update(self, experiences, epi, bs, alpha, gamma):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.uint8).unsqueeze(1)

        current_q_values = self.Q1(states).gather(1, actions)
        max_next_q_values = self.Q2(next_states).detach().max(1)[0].unsqueeze(1)
        expected_q_values = rewards + gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if epi % self.TARGET_UPDATE == 0:
            self.update_target_model()


class tabularPolicy(object):

    def __init__(self,nS,nA) -> None:
        self.Q1 = defaultdict(lambda: np.random.randint(5,10))
        self.Q2 = defaultdict(lambda: np.random.randint(5,10))
        self.nS = nS
        self.nA = nA
        self.update_target_model()
        self.TARGET_UPDATE = 5

    def update_target_model(self):
        self.Q2 = copy.deepcopy(self.Q1)

    def soft_action(self, s, epsilon):
        if np.random.uniform() < epsilon:
            return np.random.randint(0,self.nA)
        else:
            return np.argmax(np.array([ self.Q1[(s,i)] for i in range(self.nA)])) 
   
    def greedy_action(self, s, Q):
        return np.argmax(np.array([ Q[(s,i)] for i in range(self.nA)]))   
    
    def update(self,exp, epi, bs, alpha, gamma):
        for (s,a,r,sd,done) in exp:
            q = self.Q1[(s,a)]
            self.Q1[(s,a)] = q + alpha * (r + gamma * self.Q2[(sd,self.greedy_action(sd,self.Q1))] - q)
        
        if epi % self.TARGET_UPDATE == 0:
            self.update_target_model()

class DQL_Agent(object):

    def __init__(self, env, PO, ME=10000, buffer_size = 5000, batch_size = 64, epsilon = 0.1, alpha = 0.1):
        self.env = env
        self.ME = ME
        self.gamma = env.gamma
        self.alpha = alpha
        self.Q = PO(env.nS,env.nA)
        self.epsilon = epsilon
        self.action_no = env.nA
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
    def learn(self, eval_epi = 10, MS = 200, max_score = 400 ):
        best = -1e9
        best_policy = copy.deepcopy(self.Q)
        rets = []
        for epi in tqdm(range(self.ME+1)):
            s = self.env.reset()
            done = False
            steps = 0
            while not done:
                a = self.Q.soft_action(s,self.epsilon)
                #print(a)
                sd, r, done = self.env.step(a)
                self.buffer.push(s, a, r/100, sd, done)
                s = sd
                steps+=1
                if steps>MS:
                    break   
                
            if len(self.buffer) > self.batch_size:
                    experiences = self.buffer.sample(self.batch_size)
                    self.Q.update(experiences, epi, self.batch_size, self.alpha, self.gamma)
            ret = self.eval(self.Q, episodes=eval_epi, MS = MS)
            
            if best < ret:
                best = ret 
                best_policy = copy.deepcopy(self.Q) 
                print(best)
            rets.append(best)
            if best >=max_score:
                
                break
        while len(rets)<self.ME+1:
            rets.append(best)
        return best_policy,rets
    def eval(self, Q, episodes=10,  MS = 50):
        score = 0
        steps_list = []
        for episode in range(episodes):
            observation = self.env.reset()
            steps=0
            fg = 1
            while True:
                action = Q.greedy_action(observation,self.Q.Q1)
                #print(action)
                observation, reward, done = self.env.step(action)
                steps+=1
                score+=reward*fg
                fg = fg*self.gamma
                if done:
                    steps_list.append(steps)
                    break
                if steps>MS:
                    steps_list.append(steps)
                    break
        return score/episodes
        
        

def experiment(env, policy, A,E, MS, min_score, max_score, name, RUN = 5):
    N = 10000
    print(name)
    
    print("(Alpha, Epsilon): ", (A, E))
    LC = []
    best = 0
    for run in tqdm(range(RUN)):
        agent = DQL_Agent(env, policy, alpha=E, epsilon=A)
        _,lc = agent.learn(eval_epi= 10, MS =MS, max_score=max_score)
        LC.append(lc)
        best +=lc[-1]
    best/=RUN
    print("Best: ", best)
    LC = np.array(LC)
    LC= np.mean(LC, axis=0)
    std= np.std(LC, axis=0)
    #print(LC)
    plt.plot(np.arange(N+1),LC)
    plt.fill_between(np.arange(N+1), np.clip(LC - std,min_score,max_score), np.clip(LC + std,min_score,max_score), alpha=0.5)
    plt.xlabel("Episodes")
    plt.ylabel("Avg. Return")
    plt.title("Env: {} | Alpha: {} | Epsilon: {}".format(name,A,E))
    plt.savefig("saadfig/DQL_{}_{}_{}.png".format(name,A,E))
    plt.close()
                





if __name__ == "__main__":
    #env = GW687()
    #experiment(env, tabularPolicy, 0.5,0.1, 100, 0, 10, "Grid-World 687", 5)
    #env = roboBoxPushing()
    #experiment(env, tabularPolicy, 0.5,0.2, 100, -20, 100, "Robot Box-Pushing", 5)
    
    #np.random.seed(1)
    #env = CartPole()
    #experiment(env, FAPolicy,0.005,0.1, 501, 0, 500, "Cartpole", 5)
    
    np.random.seed(1)
    env = PendulumEnv(Dis = 17)
    experiment(env, FAPolicy,0.1, 0.1, 201, -2000, -200, "Pendulum", 5)
    