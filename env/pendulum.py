import numpy as np
# import gymnasium as gym
class PendulumEnv(object):
 
    def __init__(self, AS = 'D', Dis = 9):
        self.MS = 8
        self.MT = 2.0
        self.AS = AS
        self.trq = [-self.MT+i*(self.MT/(Dis-1)*2) for i in range(Dis)]
        self.nA = Dis
        self.nS = 12
        self.gamma = 1.0
        self.ts = 0
        self.s = None
        self.normal = np.array([np.pi/2, 4.0])
        self.reset()

    def step(self, a):
        g = 9.8
        dt = 0.05 
        if self.AS =='D':
            a = self.trq[a]
        r = self.reward(self.s,a)
        _thetadot = self.s[1] + (3 * g / 2 * np.sin(self.s[0]) + 3.0 * a) * dt
        _thetadot = np.clip(_thetadot, -self.MS, self.MS)
        _theta = self.s[0] + _thetadot * dt
        _theta = ((_theta + np.pi) % (2 * np.pi)) - np.pi
        self.s = np.array([_theta, _thetadot])
        self.ts +=1
        return self.state2obs(self.s), r, self.ts>=200 

    def reward(self, s, a):
        cost = s[0]**2 + 0.1*s[1]**2 + 0.001*(a**2)
        return - cost

    def reset(self,):
        self.s= np.random.uniform(low=-np.array([np.pi/2, 1.0]), high=np.array([np.pi/2, 1.0]))
        self.ts = 0
        return self.state2obs(self.s)

    def state2obs(self,s):
        x = np.clip(s/self.normal, -1,1)
        x = np.array([np.sin(x*i*np.pi) for i in range(2)]+[np.cos(x*i*np.pi) for i in range(2)]+[x for i in range(2)])
        x = x.reshape(-1)
        return x

    


