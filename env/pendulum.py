
import numpy as np

class PendulumEnv(object):
 
    def __init__(self):
        self.MS = 8
        self.MT = 2.0
        self.trq = [-self.MT+i*0.5 for i in range(9)]
        self.action_space = 9
        self.observation_space = 3
        self.ts = 0
        self.s = None
        self.reset()

    def step(self, a):
        g = 9.8
        dt = 0.05 
        a = self.trq[a]
        r = self.reward(self.s,a)
        _thetadot = self.s[1] + (3 * g / 2 * np.sin(self.s[0]) + 3.0 * a) * dt
        _thetadot = np.clip(_thetadot, -self.MS, self.MS)
        _theta = self.s[1] + _thetadot * dt
        _theta = ((_theta + np.pi) % (2 * np.pi)) - np.pi
        self.s = np.array([_theta, _thetadot])
        self.ts +=1
        return self.state2obs(self.s), r, self.ts>=200 

    def reward(self, s, a):
        cost = s[0]**2 + 0.1*s[1]**2 + 0.001*(a**2)
        return - cost

    def reset(self,):
        self.s= np.random.uniform(low=-np.array([np.pi, 1.0]), high=np.array([np.pi, 1.0]))
        self.ts = 0
        return self.state2obs(self.s)

    def state2obs(self,s):
        return np.array([np.cos(s[0]), np.sin(s[0]),np.cos(s[0]), np.sin(s[0]),np.cos(s[1]), np.sin(s[1]),np.cos(s[1]), np.sin(s[1]),  s[1]])

    