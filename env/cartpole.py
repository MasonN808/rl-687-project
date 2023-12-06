import numpy as np
import math
class CartPole(object):

    def __init__(self) -> None:
        self.gamma = 1
        self.c_s = np.array([0.0, 0.0, 0.0, 0.0])
        self.action_space = [0,1]
        self.g = 9.8 #(gravity)
        self.mc = 1.0 #(cart’s mass)
        self.mp = 0.1 #(pole’s mass)
        self.mt = self.mc + self.mp #(total mass)
        self.l = 0.5 #(pole’s length)
        self.tau = 0.02
        self.sc = 0
        self.F = [-10.0,10.0]

    def reset(self):
        self.c_s = np.array([0.0, 0.0, 0.0, 0.0])
        self.sc = 0
        return self.c_s
    
    def next_state(self, a):
        F = self.F[a]
        sin = math.sin(self.c_s[2])
        cos = math.cos(self.c_s[2])
        b = F + self.mp*self.l*(self.c_s[3]**2)*sin
        b = b/self.mt
        c = (self.g*sin - b*cos)/self.l
        c = c/(4.0/3.0-(self.mp/self.mt*cos**2))
        d = b - self.mp*self.l*c*cos/self.mt
        ns = np.array([i for i in self.c_s])
        ns[0] += self.tau*self.c_s[1]
        ns[1] += self.tau*d
        ns[2] += self.tau*self.c_s[3]
        ns[3] += self.tau*c
        return ns
    
    def isDone(self ):
        if ((self.c_s[0] < -2.4) or 
            (self.c_s[0] > 2.4) or 
            (self.c_s[2] < -math.pi/15) or 
            (self.c_s[2] > math.pi/15) or
            (self.sc>=500) ):
            return True
        return False        

    def step(self,a):
        self.c_s = self.next_state(a)
        self.sc +=1 
        return self.c_s,1.0,self.isDone()


if __name__ == "__main__":
    env = CartPole() 