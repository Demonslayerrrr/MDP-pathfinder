import random
import pprint


class RobotMdp:
    def create_grid(self,grid_size,obstacle_prob=0.3):
        
        self.grid = [[0 for i in range(grid_size)] for a in range(grid_size)]

        for i in range(grid_size):
            self.grid[i][i] = 0 

        for i in range(grid_size):
            for j in range(grid_size):
                if random.random() < obstacle_prob and i!=j:

                    self.grid[i][j] = 1

        self.grid[0][0] = 0
        
        self.grid[grid_size-1][grid_size-1] = 0
    
    def create_q_table(self):
        self.q_table = {}
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):

                self.q_table[(i,j)] = {i:0 for i in self.actions.keys()}

    def __init__(self):

        self.create_grid(5)

        self.actions = {
            "up" : (-1,0),
            "down" : (1,0),
            "left" : (0,-1),
            "right" : (0,1)
        }

        self.create_q_table()

        self.current_state = (0,0)

        self.goal = (len(self.grid)-1,len(self.grid[0])-1)
        

    def action_selection(self,state,epsilon = 0.2):
        if random.random() < epsilon: 
            action = random.choice(list(self.actions.keys()))
        else:  
            action = max(self.q_table[state], key=lambda a: self.q_table[state][a])
        return action
    
    def reset(self):
        self.current_state = (0,0)

    
    def reward(self,state):

        x,y = state

        if (x,y) == self.goal:
            return 10
        
        if self.grid[x][y] == 1:
            return -10
        
        return -1


    def update_q_table(self,state,action,reward,next_state,gamma=0.9,alpha=0.1):
        best_next_action = max(self.q_table[next_state], key=lambda action: self.q_table[next_state][action])

        td_target = reward + gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += alpha*td_error
        

    def update_current_state(self,action):
        dx,dy = self.actions[action]
        x,y = self.current_state

        self.current_state = (x+dx,y+dy)

        self.current_state = (max(0, min(self.current_state[0], len(self.grid) - 1)),
                              max(0, min(self.current_state[1], len(self.grid[0]) - 1)))
        
        return self.current_state
        
    def train(self,epochs):
        for epoch in range(epochs):
            self.reset()
            done=False
            total_reward=0

            while not done:
                current_state = self.current_state

                action = self.action_selection(current_state)

                next_state = self.update_current_state(action)

                reward = self.reward(next_state)

                self.update_q_table(current_state,action,reward,next_state)

                total_reward+=reward

                if next_state == self.goal:
                    done = True
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}/{epochs} is completed. Total reward: {total_reward}")

        pprint.pprint(self.q_table)

    def agent(self):
        self.reset()
        path = [self.current_state]
        total_reward = 0

        while self.current_state != self.goal:
            current_state = self.current_state

            best_action = max(self.q_table[current_state], key = lambda action: self.q_table[current_state][action])

            next_state = self.update_current_state(best_action)

            reward = self.reward(next_state)

            total_reward+=reward

            path.append(next_state)
        
        print(f"Agent completed the task. His path was: {path} and total reward: {total_reward}")

robot = RobotMdp()

print("GRID")
for row in robot.grid:
    print(row)

print("\nQ_TABLE")
pprint.pprint(robot.q_table)

print("\n")

robot.train(1000)

print("\n")

robot.agent()