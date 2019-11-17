from physengine import render
from physengine import norender

import threading
import pyglet
import time
import random
import collections 


class QLearn():
    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def getAction(self, actions):
        # Epsilon greedy; if generated value is less than epsilon, randomly pick an action, otherwise take the max action
        if random.uniform(0, 1) <= self.epsilon:
            return random.randrange(0, len(actions))
        else:
            return self.getMaxAction(actions)
        
    def getMaxAction(self, actions):
        max_value = max(actions)
        max_inds = [i for i, x in enumerate(actions) if x == max_value]
        return actions.index(max_value)
        
    def getUpdate(self, reward, action_taken_val, actions_newstate):
        return self.alpha * (reward + self.gamma*max(actions_newstate) - action_taken_val)
        
class TabModel():
    def __init__(self, model_type, state_space, action_space):
        # This will be used to select which algorithm to use
        if model_type is "qlearn":
            self.model_type = QLearn(0.05, 0.9, 0.1)
        self.state_space = state_space
        self.action_space = action_space
        print(self.state_space)
        # Initialize table for all states/actions, initialize to 0
        self.table = [[[[0 for x in range(action_space)] for x in range(state_space[2])] for x in range(state_space[1])] for x in range(state_space[0])]
        print("Table dimensions: ", len(self.table), " x ", len(self.table[0]), " x ", len(self.table[0][0]), " x ", len(self.table[0][0][0]))
        
    def getAction(self):
        return self.model_type.getAction(self.getActionsForState(self.state))
        
    def getActionsForState(self, state):
        return self.table[state[0]][state[1]][state[2]]
        
    def setState(self, state):
        print(state)
        self.state = state
        
    def update(self, reward, action_taken, newstate):
        action_taken_val = self.getActionsForState(self.state)[action_taken]
        self.table[self.state[0]][self.state[1]][self.state[2]][action_taken] += self.model_type.getUpdate(reward, action_taken_val, self.getActionsForState(newstate))
        self.state = newstate

class Learner():
    def __init__(self, winwidth, winheight, rendered):
        # Whether or not training should be rendered
        self.rendered = rendered
        self.dt = 0.02
        if self.rendered:
            self.engine = render.PhysWin(winwidth, winheight)
        else:
            self.engine = norender.PymunkSpaceNoRender(winwidth, winheight)
        
        # Attributes from the engine to calculate
        # TODO: Replace this statement with an actual model
        state_space = self.engine.getStateSpace()
        action_space = self.engine.getActionSpace()
            
        self.model = TabModel("qlearn", state_space, action_space)
        
    def start(self):
        if self.rendered:
            thread = threading.Thread(target = self.control_rendered)
            thread.start()
            pyglet.app.run()

        else:
            self.shot_count = 0
            self.ep_count = 0
            self.avg_shot_count = 0
            self.last10count = collections.deque(range(0,9))
            # Get the initial S
            net_x, net_y, ball_x = self.engine.getCoords()
            self.model.setState(self.engine.mapValsToState(net_x, net_y, ball_x))
            while True:
                self.control_norendered()
        
    def control_rendered(self):
        # This is blocking, so the thread will wait until the Physwin is done initializing
        render.done_queue.get()
        # Get the initial S
        net_x, net_y, ball_x = self.engine.getCoords()
        self.model.setState(self.engine.mapValsToState(net_x, net_y, ball_x))
        while True:
            actionidx = self.model.getAction()
            action = self.engine.mapActionToVals(actionidx)
            render.job_queue.put(action)
            done = render.done_queue.get()
            reward = 0
            if action.action_type == "shoot":
                if done:
                    reward = 1
                else:
                    reward = -1
            net_x, net_y, ball_x = self.engine.getCoords()
            self.model.update(reward, actionidx, self.engine.mapValsToState(net_x, net_y, ball_x))

            
    def control_norendered(self):
        actionidx = self.model.getAction()
        action = self.engine.mapActionToVals(actionidx)
        if action.action_type == "shoot":
            #print("Shooting ", action.velocity, " at angle ", action.angle)
            made_basket = self.engine.shoot(action, self.dt)
            if made_basket:
                reward = 1
                print("Basket made!")
                self.last10count.popleft()
                self.last10count.append(self.shot_count)
                avg_shot_count = sum(self.last10count)/len(self.last10count)
                print("Average of shots per episode in last 10 episodes: ", avg_shot_count)
                self.shot_count = 0
                
            else:
                reward = -1
                #print("Shot count: " , self.shot_count)
                self.shot_count += 1
            net_x, net_y, ball_x = self.engine.getCoords()
            self.model.update(reward, actionidx, self.engine.mapValsToState(net_x, net_y, ball_x))
        else:
            self.engine.move(action.action_type)
            # Reward of 0 for a movement
            net_x, net_y, ball_x = self.engine.getCoords()
            self.model.update(0, actionidx, self.engine.mapValsToState(net_x, net_y, ball_x))

            
    def getAction(self):
        return self.model.getAction(bball_x, bball_y, net_x, net_y)
        
    
if __name__ == "__main__":
    random.seed(3)
    learner = Learner(500, 500, False)
    learner.start()
