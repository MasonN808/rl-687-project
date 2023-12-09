from env.cartpole import CartPole
from env.pendulum import PendulumEnv
from env.gridworld import GW687
from env.boxpushing import roboBoxPushing
import numpy as np
import copy 
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    ret = []
    
    for i in range(N):
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
        ret.append(best)
        if best>= max_score:
            break
    while len(ret)<N:
        ret.append(best)
    return best_policy,ret
    


def experiment(env, policy,min_score, max_score, name, RUN = 5):
    Ts = [1,5,10,20,40]
    Eval = [1,5,10,20,40]
    N = 10000
    print(name)
    for T in Ts:
        for E in Eval:
            print("(T, EVAL): ", (T,E))
            LC = []
            best = 0
            for run in tqdm(range(RUN)):
                _,lc = Simulated_Anneling(env, policy, N = N, eval=E, T=T, max_score=max_score)
                LC.append(lc)
                best +=lc[-1]
            best/=RUN
            print("Best: ", best)
            LC = np.array(LC)
            LC= np.mean(LC, axis=0)
            std= np.std(LC, axis=0)
            #print(LC)
            plt.plot(np.arange(N)*E,LC)
            plt.fill_between(np.arange(N)*E, np.clip(LC - std,min_score,max_score), np.clip(LC + std,min_score,max_score), alpha=0.5)
            plt.xlabel("Episodes")
            plt.ylabel("Avg. Return")
            plt.title("Env: {} | T: {} | n_episodes: {}".format(name,T,E))
            plt.savefig("saadfig/{}_{}_{}.png".format(name,T,E))
            plt.close()
                





if __name__ == "__main__":
    #env = CartPole()
    #experiment(env,cartpolePolicy, 0,500, "Cartpole", 20)
    env = PendulumEnv("D",Dis = 161)
    experiment(env,pendulamPolicyD, -2000,0, "Pendulum", 5)
    #env = GW687()
    #experiment(env, GW687Policy, -10, 10, "Grid-World 687", 20)
    #env = roboBoxPushing()
    #experiment(env, RBPPolicy, -500, 70, "Robot Box-Pushing", 20)
    