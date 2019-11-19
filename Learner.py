from physengine import render
from physengine import norender
import matplotlib.pyplot as plt
from tqdm import tqdm

import threading
import pyglet
import time
import datetime
import random
import collections 
import pickle
import os
import argparse


class TabModel():
    def __init__(self, engine, controller, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.engine = engine
        self.controller = controller
        self.dt = 0.02
        self.state_space = self.engine.getStateSpace()
        self.action_space = self.engine.getActionSpace()
        # Initialize table for all states/actions, initialize to 0
        self.table = [[[[0 for x in range(self.action_space)] for x in range(self.state_space[2])] for x in range(self.state_space[1])] for x in range(self.state_space[0])]
        #self.loadTable("testtable.pkl")
        print("Table dimensions: ", len(self.table), " x ", len(self.table[0]), " x ", len(self.table[0][0]), " x ", len(self.table[0][0][0]))
        
    def getAction(self, state):
        actions = self.getActionsForState(state)
        return self.getActionEGreedy(actions)
        
    def getActionsForState(self, state):
        return self.table[state[0]][state[1]][state[2]]
        
    def getState(self):
        net_x, net_y, ball_x = self.engine.getCoords()
        return self.engine.mapValsToState(net_x, net_y, ball_x)
        
    def getActionEGreedy(self, actions):
        # Epsilon greedy; if generated value is less than epsilon, randomly pick an action, otherwise take the max action
        if random.uniform(0, 1) <= self.epsilon:
            return random.randrange(0, len(actions))
        else:
            return self.getMaxAction(actions)
        
    def getMaxAction(self, actions):
        max_value = max(actions)
        #max_inds = [i for i, x in enumerate(actions) if x == max_value]
        return actions.index(max_value)
        
    def doShoot(self, action):
        if self.controller:
            missed_by = self.controller.shoot(action)
        else:
            missed_by = self.engine.shoot(action, self.dt)
        if missed_by is 0:
            reward = 1
            basket_made = True
        else:
            # Make this negative, as a miss should be a negative reward
            reward = -missed_by
            basket_made = False
        return basket_made, reward
        
    def update(self, reward, actionidx, action_taken_val, next_action_val):
        updt = self.alpha * (reward + self.gamma*next_action_val - action_taken_val)
        self.table[self.state[0]][self.state[1]][self.state[2]][actionidx] += updt
        
    def saveTable(self, filename):
        pickle.dump(self.table, open(filename, "wb"))
    
    def loadTable(self, filename):
        self.table = pickle.load(open(filename, "rb"))
        
class QLearn(TabModel):
    def __init__(self, engine, controller, alpha, gamma, epsilon):
        TabModel.__init__(self, engine, controller, alpha, gamma, epsilon)
        
    def runEpisode(self):
        self.shots_taken = 0
        self.state = self.getState()
        basket_made = False
        while basket_made is False:
            actionidx = self.getAction(self.state)
            action_taken_val = self.getActionsForState(self.state)[actionidx]
            action = self.engine.mapActionToVals(actionidx)
            if action.action_type == "shoot":
                basket_made, reward = self.doShoot(action)
                self.shots_taken += 1
            else:
                if self.controller:
                    self.controller.move(action.action_type)
                else:
                    self.engine.move(action.action_type)
                # Reward of 0 for a movement
                reward = 0
            
            new_state = self.getState()
            self.update(reward, actionidx, action_taken_val, max(self.getActionsForState(new_state)))
            self.state = new_state
        return self.shots_taken
                    
class Sarsa(TabModel):
    def __init__(self, engine, controller, alpha, gamma, epsilon):
        TabModel.__init__(self, engine, controller, alpha, gamma, epsilon)
        
    def runEpisode(self):
        self.shots_taken = 0
        self.state = self.getState()
        actionidx = self.getAction(self.state)
        action_taken_val = self.getActionsForState(self.state)[actionidx]
        basket_made = False
        while basket_made is False:
            action = self.engine.mapActionToVals(actionidx)
            if action.action_type == "shoot":
                basket_made, reward = self.doShoot(action)
                self.shots_taken += 1
            else:
                if self.controller:
                    self.controller.move(action.action_type)
                else:
                    self.engine.move(action.action_type)
                # Reward of 0 for a movement
                reward = 0
            
            new_state = self.getState()
            new_actionidx = self.getAction(new_state)
            new_action_val = self.getActionsForState(new_state)[new_actionidx]
            self.update(reward, actionidx, action_taken_val, new_action_val)

            self.state = new_state
            actionidx = new_actionidx
            action_taken_val = new_action_val
        return self.shots_taken
            

class Learner():
    def __init__(self, winwidth, winheight, model_type, rendered):
        self.model_type = model_type
        # Whether or not training should be rendered
        self.rendered = rendered
        controller = None
        if self.rendered:
            engine = render.PhysWin(winwidth, winheight)
            controller = render.PhysWinController()
        else:
            engine = norender.PymunkSpaceNoRender(winwidth, winheight)
        
        if self.model_type == "qlearn":
            self.model = QLearn(engine, controller, 0.025, 0.75, 0.1)
        elif self.model_type == "sarsa":
            self.model = Sarsa(engine, controller, 0.025, 0.75, 0.1)
        
    def start(self, episodes, graph):
        self.graph = graph
        if self.rendered:
            thread = threading.Thread(target = self.control, args=([episodes]))
            thread.start()
            pyglet.app.run()

        else:
            self.control(episodes)
        
    def control(self, episodes):
        shots_taken_graph = []
        pbar = tqdm(total=episodes)
        for i in range(episodes):
            if i % 100 is 0:
                pbar.update(100)
            shots_taken = self.model.runEpisode()
            shots_taken_graph.append(shots_taken)
        pbar.close()
        graphfilename, tablefilename = self.saveListAndTable(shots_taken_graph)
        self.plotGraph(shots_taken_graph, graphfilename)
        
        
    def plotGraph(self, shots, graphfilename):
        eps = range(len(shots))

        fig, ax = plt.subplots()
        ax.plot(eps, shots)

        ax.set(xlabel='Episode', ylabel='Shots Taken',
               title='Shots Taken Per Episode')
        ax.grid()

        print("Saving " + graphfilename + ".png...")
        fig.savefig(graphfilename + ".png")
        if self.graph:
            plt.show()
    
    def saveListAndTable(self, shots_taken_graph):
        dirname = os.path.dirname(os.path.abspath(__file__))
        ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_").replace(":", "-")
        graphfilename = dirname + "/data/" + ts + "_graph-" + self.model_type + ".pkl"
        tablefilename = dirname + "/data/" + ts + "_table-" + self.model_type + ".pkl"
        print("Saving " + graphfilename + "...")
        pickle.dump(shots_taken_graph, open(graphfilename, "wb"))
        print("Saving " + tablefilename + "...")
        self.model.saveTable(tablefilename)
        return graphfilename, tablefilename

        
    def getAction(self):
        return self.model.getAction(bball_x, bball_y, net_x, net_y)
        
    
if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
            
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether or not to display the graph of the results")
    parser.add_argument("--eps", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--algo", type=str, default="sarsa", choices=["qlearn","sarsa"], help="Which RL algorithm to use")
    parser.add_argument("--render", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether or not to render the game")
    args = parser.parse_args()
    graph = args.graph
    eps = args.eps
    algo = args.algo
    rendered = args.render

    random.seed(3)
    learner = Learner(500, 500, algo, rendered)
    learner.start(eps, graph)
