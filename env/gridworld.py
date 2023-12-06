import numpy as np

class GW687(object):

    def __init__(self,gamma = 0.9) -> None:
        self.r, self.c = 5,5
        self.grid = np.array([
            [4,0,0,0,0],
            [0,0,0,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,2,0,3]
        ])
        self.states, self.itos, self.stoi = self.get_states()
        self.actions = [0,1,2,3]
        self.na = 4
        self.ns = 25
        self.cs = 0
        self.gamma = gamma
        self.R = self.get_rewards()
        self.T = self.get_transitions()
        self.VS = np.array([[4.0187e+00, 4.5548e+00, 5.1575e+00, 5.8336e+00, 6.4553e+00],
                            [4.3716e+00, 5.0324e+00, 5.8013e+00, 6.6473e+00, 7.3907e+00],
                            [3.8672e+00, 4.3900e+00, 0.0000e+00, 7.5769e+00, 8.4637e+00],
                            [3.4182e+00, 3.8319e+00, 0.0000e+00, 8.5738e+00, 9.6946e+00],
                            [2.9977e+00, 2.9309e+00, 6.0733e+00, 9.6946e+00, 0.0000e+00]])
        self.PI = np.array([
                            [3,3,3,1,1],
                            [3,3,3,1,1],
                            [0,0,3,1,1],
                            [0,0,3,1,1],
                            [0,0,3,3,3]
                        ])

    def reset(self):
        self.cs = np.random.choice([i for i in range(25) if i not in [12,17,24]])
        return self.cs

    def get_states(self):
        states = []
        itos = {}
        stoi = {}
        cnt = 0
        for r in range(self.r):
            for c in range(self.c):
                states.append((r,c))
                itos[cnt] = (r,c)
                stoi[(r,c)] = cnt
                cnt+=1
        return states, itos, stoi 
    
    def bounce(self,r,c):
        if r>4 or r<0 or c>4 or c<0:
            return True
        elif (r,c) == (2,2) or (r,c) == (3,2):
            return True
        return False
                
    def get_transitions(self):
        T = np.zeros((self.ns,self.na,self.ns))
        UD = [-1,1,0,0]
        LR = [0,0,-1,1]
        for s in range(self.ns):
            for a in range(self.na):
                cs = s
                left = None
                right = None
                sd = None
                r,c = self.itos[s]
             
                r_ = r+UD[a]
                c_ = c+LR[a]
              
                if self.bounce(r_,c_):
                    sd = s
                else:
                    sd = self.stoi[(r_,c_)]
                if a<2:
                    if self.bounce(r,c+1):
                        left = s
                    else:
                        left = self.stoi[(r,c+1)]
                    if self.bounce(r,c-1):
                        right = s
                    else:
                        right = self.stoi[(r,c-1)]
                else:
                    if self.bounce(r+1,c):
                        left = s
                    else:
                        left = self.stoi[(r+1,c)]
                    if self.bounce(r-1,c):
                        right = s
                    else:
                        right = self.stoi[(r-1,c)]
                
                T[s,a,cs]+=0.1
                
                T[s,a,sd]+=0.8
                T[s,a,left]+=0.05
                T[s,a,right]+=0.05
                T[s,a]/=np.sum(T[s,a])
        T[24,:,:] = 0
        T[24,:,24] = 1.0
        T[12,:,:] = 0
        T[12,:,12] = 1.0
        T[17,:,:] = 0
        T[17,:,17] = 1.0
        return T
        

    def get_rewards(self):
        R = np.zeros((self.ns,self.na,self.ns))
        for s in range(self.ns):
            for a in range(self.na):
                for sd in range(self.ns):
                    if self.itos[sd] == (4,2):
                        R[s,a,sd] = -10
                    elif self.itos[s] in [(4,4),(2,2),(3,2)]:
                        R[s,a,sd] = 0
                    elif self.itos[sd] == (4,4):
                        R[s,a,sd] = 10
                    else:
                        R[s,a,sd] = 0  
        return R
    
    def step(self,a):
        sd = np.random.choice(np.arange(self.ns), p = self.T[self.cs,a,:])
        r = self.R[self.cs,a,sd]
        done = (self.itos[sd]== (4,4))
        self.cs = sd
        return sd,r,done
