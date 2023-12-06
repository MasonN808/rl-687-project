import numpy as np
from env.gridworld import GW687
from tqdm import tqdm
from Draw import drawPolicy
import matplotlib.pyplot as plt

def choose_action(state, Q, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice([0, 1, 2, 3])  
    else:
        return np.argmax(Q[state])  

def SARSA(env, alpha, epsilon = 0.02, delta = 1e-3):
    Q = np.ones((5, 5, 4))*5  
    Q[4,4] = 0
    Q[2,2] = 0
    Q[3,2] = 0
    epi_count = 0
    step_count = 0
    epi_list = []
    step_list = []
    verror = []
    while True:
        epi_count+=1
        state = env.reset()
        assert state not in [12,17,24]
        state = env.itos[state]
        action = choose_action(state, Q, epsilon)
        d =  0
        while True:
            next_state, reward, done = env.step(action)
            step_count+=1
            step_list.append(step_count)
            epi_list.append(epi_count)
            next_state = env.itos[next_state]
            next_action = choose_action(next_state, Q, epsilon)
            pv = Q[state][action]
            Q[state][action] += alpha * (reward + env.gamma * Q[next_state][next_action] - Q[state][action])
            d = max(d,abs(Q[state][action]-pv))
            state = next_state
            action = next_action
            if done:
                break
        V = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                m = np.max(Q[i,j])
                tot = 0
                for a in range(4):
                    if Q[i,j,a] == m:
                        V[i,j]+=Q[i,j,a]
                        tot+=1
                    else:
                        V[i,j]+=Q[i,j,a]*epsilon
                        tot+=epsilon
                V[i,j]/=tot

        verror.append(np.mean((V - env.VS)**2))
        if d<delta:
            break
    return Q,step_list,epi_list,verror


if __name__ == "__main__":
    Q = np.zeros((5,5,4))
    np.random.seed(1)
    EC = []
    VER = []
    run = 20
    for i in tqdm(range(run)):
        env = GW687()
        q,sc,ec,ver = SARSA(env,0.2)
        print(len(ver))
        EC.append(ec)
        VER.append(ver)
        Q+=q
        #E.append(e)
    Q/=run
    _EC = []
    min_len = np.min([len(EC[i]) for i in range(run)])
    max_len = np.max([len(VER[i]) for i in range(run)])
    
    for i in range(run):
        _EC.append(EC[i][:min_len])
    for i in range(run):
        for j in range(len(VER[i]),max_len):
            VER[i].append(VER[i][-1])
    EC = np.array(_EC)
    VER = np.array(VER)
    print(EC.shape,VER.shape)
    mean_EC= np.mean(EC, axis=0)
    std_EC= np.std(EC, axis=0)
    mean_VER= np.mean(VER, axis=0)[:1000]
    std_VER= np.std(VER, axis=0)[:1000]
    plt.plot(np.arange(min_len),mean_EC)
    plt.fill_between(np.arange(min_len), np.clip(mean_EC - std_EC,0,3000), np.clip(mean_EC + std_EC,0,3000), alpha=0.5)
    plt.xlabel("Total number of actions taken")
    plt.ylabel("Number of episodes completed")
    plt.title("SARSA: Actions vs. Episodes")
    plt.savefig("SARSA:AE")
    plt.close()
    plt.plot(np.arange(1000),mean_VER)
    plt.fill_between(np.arange(1000), np.clip(mean_VER - std_VER,0,8000), np.clip(mean_VER + std_VER, 0,8000), alpha=0.5)
    plt.xlabel("Number of episodes")
    plt.ylabel("MSE")
    plt.title("SARSA: Episodes vs. MSE")
    plt.savefig("SARSA:ME")
    plt.close()
    drawPolicy(np.argmax(Q, axis=2),"SARSA","SARSA Policy", {'G':[(4,4)],'E':[(2,2),(3,2)]})
