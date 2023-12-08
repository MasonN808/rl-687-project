import numpy as np
from env.cartpole import CartPole
from env.pendulum import PendulumEnv
from env.gridworld import GW687
from env.boxpushing import roboBoxPushing
from collections import defaultdict,  deque
import copy
import random

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

    def __init__(self,nS,nA) -> None:
        self.Q1 = np.random.rand(nA,nS)-0.5
        self.Q2 = copy.deepcopy(self.Q1)
        self.nS = nS
        self.nA = nA
        print(self.Q1.shape)

    def soft_action(self, s, epsilon):
        if np.random.uniform() < epsilon:
            return np.random.randint(0,self.nA)
        else:
            return np.argmax(self.Q1.dot(s)+self.Q2.dot(s)) 
   
    def greedy_action(self, s, Q):
        return np.argmax(Q.dot(s))   
    
    def q_val(self, s, a, Q):
        return Q[a,:].dot(s)
    
    def update(self, exp, bs, alpha, gamma):
        G1 = np.zeros((self.nA,self.nS))
        G2 = np.zeros((self.nA,self.nS))
        bs1 = 0
        bs2 = 0
        for s, a, r, sd, done in exp:
            if np.random.uniform() < 0.5:
                q = self.q_val(s, a, self.Q1)
                a_max = self.greedy_action(sd, self.Q2)
                target = r + gamma * self.q_val(sd, a_max, self.Q2) * (not done)
                G1[a,:] -= alpha  * (target - q) * s
                bs1+=1
            else:
                q = self.q_val(s, a, self.Q2)
                a_max = self.greedy_action(sd, self.Q1)
                target = r + gamma * self.q_val(sd, a_max, self.Q1) * (not done)
                G2[a,:] -= alpha * (target - q) * s
                bs2+=1
        self.Q1+=G1/bs1
        self.Q2+=G2/bs2
            


class tabularPolicy(object):

    def __init__(self,nS,nA) -> None:
        self.Q1 = defaultdict(lambda: np.random.randint(5,10))
        self.Q2 = copy.deepcopy(self.Q1)
        self.nS = nS
        self.nA = nA


    def soft_action(self, s, epsilon):
        if np.random.uniform() < epsilon:
            return np.random.randint(0,self.nA)
        else:
            return np.argmax(np.array([ self.Q1[(s,i)]+self.Q2[(s,i)] for i in range(self.nA)])) 
   
    def greedy_action(self, s, Q):
        return np.argmax(np.array([ Q[(s,i)] for i in range(self.nA)]))   
    
    def update(self,exp,bs, alpha, gamma):
        for (s,a,r,sd,done) in exp:
            if np.random.uniform() < 0.5: 
                    q = self.Q1[(s,a)]
                    self.Q1[(s,a)] = q + alpha * (r + gamma * self.Q2[(sd,self.greedy_action(sd,self.Q1))] - q)
            else:
                
                    q = self.Q2[(s,a)]
                    self.Q2[(s,a)] = q + alpha * (r + gamma * self.Q1[(sd,self.greedy_action(sd,self.Q2))] - q)
        


class DQL_Agent(object):

    def __init__(self, env, PO, ME=10000, buffer_size = 5000, batch_size = 64, epsilon = 0.05, alpha = 0.1):
        self.env = env
        self.ME = ME
        self.gamma = env.gamma
        self.alpha = alpha
        self.Q = PO(env.nS,env.nA)
        self.epsilon = epsilon
        self.action_no = env.nA
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
    
    def learn(self, eval_epi = 10, MS = 200, verbos = 10 ):
        best = -1e9
        best_policy = copy.deepcopy(self.Q)
        for epi in range(self.ME+1):
            s = self.env.reset()
            done = False
            steps = 0
            while not done:
                a = self.Q.soft_action(s,self.epsilon)
                sd, r, done = self.env.step(a)
                self.buffer.push(s, a, r, sd, done)
                s = sd
                steps+=1
                if steps>MS:
                    break   
                
            if len(self.buffer) > self.batch_size:
                    experiences = self.buffer.sample(self.batch_size)
                    self.Q.update(experiences,self.batch_size, self.alpha, self.gamma)
            ret = self.eval(self.Q, episodes=eval_epi, MS = MS)
            
            if best < ret:
                best = ret 
                best_policy = copy.deepcopy(self.Q) 
                print(best)
        self.eval(best_policy, episodes=eval_epi, MS = MS)
    
    def eval(self, Q, episodes=10,  MS = 50):
        score = 0
        steps_list = []
        for episode in range(episodes):
            observation = self.env.reset()
            steps=0
            while True:
                action = Q.greedy_action(observation,self.Q.Q1)
                observation, reward, done = self.env.step(action)
                steps+=1
                score+=reward
                if done:
                    steps_list.append(steps)
                    break
                if steps>MS:
                    steps_list.append(steps)
                    break
        return score/episodes
        
        

if __name__ == "__main__":
    #env = GW687()
    #agent = DQL_Agent(env,tabularPolicy,buffer_size =500, alpha=0.5)
    #agent.learn()

    env = roboBoxPushing()
    agent = DQL_Agent(env,tabularPolicy,buffer_size =500, epsilon=0.1, alpha=0.5)
    agent.learn(MS = 500, eval_epi= 10)
    
    np.random.seed(1)
    env = CartPole()
    env.gamma = 0.9
    agent = DQL_Agent(env, FAPolicy, alpha=0.05)
    agent.learn(eval_epi= 20,verbos=1, MS =501)
   
    
    env = PendulumEnv(Dis = 161)
    env.gamma = 0.9
    agent = DQL_Agent(env, FAPolicy, alpha=0.5)
    agent.learn(eval_epi= 20,verbos=10, MS =201)