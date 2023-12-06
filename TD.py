import numpy as np
from env.gridworld import GW687
from tqdm import tqdm
from Draw import drawValue
def TD0(env, alpha, delta = 1e-3):
    V = np.random.rand(env.grid.shape[0],env.grid.shape[1])*5
    V[4,4] = 0
    V[2,2] = 0
    V[3,2] = 0
    cnt = 0
    while True:
        cnt+=1
        state = env.reset()
        assert state not in [12,17,24]
        state = env.itos[state]
        d =  0
        
        while True:
            action = env.PI[state]
            next_state, reward, done = env.step(action)
            next_state = env.itos[next_state]
            pv = V[state]
            V[state] += alpha * (reward + env.gamma * V[next_state] - V[state])
            d = max(d,abs(V[state]-pv))
            state = next_state
            if done:
                break
        if d<delta:
            break
    return V,cnt



if __name__ == "__main__":
    V = np.zeros((5,5))
    np.random.seed(1)
    E = []
    for i in tqdm(range(50)):
        env = GW687()
        v,e = TD0(env,0.2)
        V+=v
        E.append(e)
    V/=50
    env = GW687()
    print(V)
    print(env.VS)
    print(np.max(np.abs(V-env.VS)))
    print(np.mean(E))
    print(np.std(E))
    drawValue(np.round(V, 4),"TD","TD-Learning Policy Evaluation")



        
    