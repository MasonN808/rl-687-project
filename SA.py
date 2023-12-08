from env.cartpole import CartPole
from env.pendulum import PendulumEnv
from env.gridworld import GW687
from env.boxpushing import roboBoxPushing
import numpy as np
import copy 
from tqdm import tqdm

class GW687Policy(object):

    def __init__(self,sl=25, al = 4) -> None:
        self.param = np.random.randint(low=0,high=4,size=(sl,))
        self.maxlen = 40
    
    def mutate(self):
        self.param[np.random.randint(self.param.shape[0])] = np.random.randint(4)

    def action(self,s):
        return  self.param[s]
    
class RBPPolicy(object):

    def __init__(self,sl=1250, al = 5) -> None:
        self.param = np.random.randint(low=0,high=5,size=(sl,))
        self.maxlen = 1000
    
    def mutate(self):
        self.param[np.random.randint(self.param.shape[0])] = np.random.randint(5)

    def action(self,s):
        return  self.param[s]
        

class cartpolePolicy(object):
    
    def __init__(self, sl = 20, al = None) -> None:
        self.param = np.random.rand(sl)-0.5
        self.maxlen = 501
    def mutate(self):
        self.param = self.param*(0.8)+ (np.random.rand(self.param.shape[0])-0.5)*0.2
    def action(self,s):
        threshold =  self.param.dot(s)
        if threshold > 0:
            return 1
        else:
            return 0
        
class pendulamPolicy(object):
    
    def __init__(self, sl = 9, al = 9) -> None:
        self.param = np.random.rand(sl)-0.5
        self.maxlen = 201
    def mutate(self):
        self.param = self.param*(0.8)+ (np.random.rand(self.param.shape[0])-0.5)*0.2
    def action(self,s):
        x =  self.param.dot(s)
        x = 1/(1 + np.exp(-x+1e-9))
        return x*4-2
    
class pendulamPolicyD(object):
    
    def __init__(self, sl = 9, al = 9) -> None:
        self.param = np.random.rand(al,sl)-0.5
        self.maxlen = 201
    def mutate(self):
        self.param = self.param*(0.8)+ (np.random.rand(self.param.shape[0],self.param.shape[1])-0.5)*0.2
    def action(self,s):
        x =  self.param.dot(s)
        return np.argmax(x)
    
def evalPolicy(env, policy, N):
    rets = []
    for i in range(N):
        ret = 0
        gamma = 1
        c_s = env.reset()
        itr = 0
        done = False
        while not done:
            action = policy.action(c_s)
            itr+=1
            n_s,r,done = env.step(action)
            c_s = n_s   
            ret += gamma*r
            gamma = gamma*env.gamma
            if itr>policy.maxlen:
                break
        rets.append(ret)
        return np.mean(rets)
    
def Simulated_Anneling(env, PO, N = 100, eval =5, T = 25, max_score = 500):
    best = -1e9
    current_performacne = best
    best_policy = None
    current_policy = PO(env.nS,env.nA)
    for i in tqdm(range(N)):
        env.reset()
        new_policy = copy.deepcopy(current_policy)
        new_policy.mutate()
        r = evalPolicy(env,new_policy,eval)
        if r >= current_performacne: 
            current_policy = copy.deepcopy(new_policy)
            current_performacne = r
        elif np.random.rand()<np.e**(-T*(current_performacne-r)/(current_performacne+1)):
            current_policy = copy.deepcopy(new_policy)
            current_performacne = r
        if best < current_performacne:
            best = current_performacne
            best_policy = copy.deepcopy(current_policy)
            print(i,best)
        if best>= max_score:
            break
    print("Best: ", best)
    return best_policy
    



if __name__ == "__main__":
    #env = CartPole()
    #Simulated_Anneling(env,cartpolePolicy, 1000, 500)
    #env = gymP()
    #Simulated_Anneling(env, pendulamPolicy, N = 1000, T = 20, max_score=200)
    #env = PendulumEnv("C")
    #Simulated_Anneling(env, pendulamPolicy, N = 1000, T = 15, max_score=200)
    #env = PendulumEnv("D",Dis = 161)
    #Simulated_Anneling(env, pendulamPolicyD, N = 10000, T = 15, max_score=200)
    #env = GW687()
    #Simulated_Anneling(env, GW687Policy, N = 10000, T = 5, max_score=7)
    env = roboBoxPushing()
    Simulated_Anneling(env, RBPPolicy, N = 100000,eval = 625, T = 5, max_score=100)
