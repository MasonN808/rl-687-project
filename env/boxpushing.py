import numpy as np

class roboBoxPushing:
    def __init__(self):
        self.grid_size = 5
        self.robot_position = (4, 0)  # bottom left corner
        self.goal_position = (4, 4)   # lower right corner
        self.box_position = self._random_box_position()
        self.has_box = False
        self.actions = [0, 1, 2, 3, 4]
        self.nS = 1250
        self.nA = 5
        self.states, self.itos, self.stoi = self.get_states()
        self.gamma = 0.9
        

    def get_states(self):
        states = []
        itos = {}
        stoi = {}
        cnt = 0
        for r1 in range(self.grid_size):
            for c1 in range(self.grid_size):
                for r2 in range(self.grid_size):
                    for c2 in range(self.grid_size):
                        for f in range(2):
                            states.append((r1,c1,r2,c2,f))
                            itos[cnt] = (r1,c1,r2,c2,f)
                            stoi[(r1,c1,r2,c2,f)] = cnt
                            cnt+=1
        return states, itos, stoi 

    def _random_box_position(self):
        while True:
            bx, by = np.random.randint(0, self.grid_size, size=2)
            if (bx, by) != self.robot_position and (bx, by) != self.goal_position:
                return (bx, by)

    def step(self, action):
        if action != 4:
            if np.random.rand() < 0.05:
                return self.get_state(),self.reward(action),self.is_goal_reached()

            if np.random.rand() < 0.10 :
                action = np.random.choice([0,1,2,3])

        x, y = self.robot_position
        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and y < self.grid_size - 1:
            y += 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.grid_size - 1:
            x += 1
        elif action == 4 and self.robot_position == self.box_position:
            self.has_box = True

        self.robot_position = (x, y)
        return self.get_state(),self.reward(action),self.is_goal_reached()

    def get_state(self):
        return self.stoi[(*self.robot_position, *self.box_position, self.has_box)]

    def get_factored_state(self):
        return (*self.robot_position, *self.box_position, self.has_box)

    def is_goal_reached(self):
        return self.robot_position == self.goal_position and self.has_box

    def reward(self,a):
        return 100 if self.is_goal_reached() else -1

    def reset(self):
        self.robot_position = (4, 0)
        self.box_position = self._random_box_position()
        self.has_box = False
        return self.get_state()

if __name__ == "__main__":
    grid_world = roboBoxPushing()
    state = grid_world.reset()
    done = False
    total_reward = 0
    steps = 0 
    while not done:
        action = np.random.choice(grid_world.actions)  # Replace with your action selection logic
        print(state,action)
        state,reward,done = grid_world.step(action)
        total_reward += reward
        steps +=1
    print("Total Reward:", total_reward)
    print("Total steps:", steps)
