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
from math import exp
import numpy as np
import configparser

import tensorflow
from keras.models import Sequential
from keras.models import Model as kModel
from keras.layers import Input, Dense
from keras.optimizers import Adam

class Model():
    def __init__(self, engine, controller, alpha, gamma, epsilon, distReward):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.distReward = distReward

        self.engine = engine
        self.controller = controller
        self.dt = 0.02
        self.state_space = self.engine.getStateSpace()
        self.action_space = self.engine.getActionSpace()
        
    def getState(self):
        net_x, net_y, ball_x = self.engine.getCoords()
        return self.engine.mapValsToState(net_x, net_y, ball_x)
        
    def getActionEGreedy(self, actions):
        # Epsilon greedy; if generated value is less than epsilon, randomly pick an action, otherwise take the max action
        if random.uniform(0, 1) <= self.epsilon:
            idx = random.randrange(0, len(actions))
            return actions[idx], idx
        else:
            return self.getMaxAction(actions)
        
    def getMaxAction(self, actions):
        max_value = max(actions)
        #max_inds = [i for i, x in enumerate(actions) if x == max_value]
        return max_value, actions.index(max_value)
        
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
            if self.distReward:
                reward = -missed_by
            else:
                reward = -1
            basket_made = False
        return basket_made, reward
        
class ApproxModel(Model):
    def __init__(self, engine, controller, alpha, gamma, epsilon, distReward):
        Model.__init__(self, engine, controller, alpha, gamma, epsilon, distReward)

        self.nn = Sequential()
        self.nn.add(Dense(200, batch_input_shape=(1,len(self.state_space)), activation='sigmoid'))
        self.nn.add(Dense(100, activation='sigmoid'))
        self.nn.add(Dense(self.action_space, activation='linear'))
        opt = Adam(learning_rate=self.alpha)
        self.nn.compile(loss='mse', optimizer=opt, metrics=['mae'])

class SGSarsa(ApproxModel):
    def __init__(self, engine, controller, alpha, gamma, epsilon, distReward):
        ApproxModel.__init__(self, engine, controller, alpha, gamma, epsilon, distReward)

    def runEpisode(self):
        self.shots_taken = 0
        self.state = self.getState()
        action_taken_val, actionidx, actions = self.getAction(self.state)
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
                # Reward of -0.1 for movement
                reward = -0.1
            
            new_state = self.getState()
            if basket_made is True:
                target = reward
                actions[actionidx] = target
                self.nn.fit(np.array([self.state,]), actions.reshape(-1, len(actions)), epochs=1, verbose=0)
                return self.shots_taken

            new_action_val, new_actionidx, new_actions = self.getAction(new_state)
            target = reward + self.gamma*new_action_val
            actions[actionidx] = target
            self.nn.fit(np.array([self.state,]), actions.reshape(-1, len(actions)), epochs=1, verbose=0)
            #self.update(reward, actionidx, action_taken_val, new_action_val)

            self.state = new_state
            actionidx = new_actionidx
            action_taken_val = new_action_val
        return self.shots_taken

    def getAction(self, state):
        actions = self.nn.predict(np.array([state,]))[0]
        val, idx = self.getActionEGreedy(actions.tolist())
        return val, idx, actions
        

class TabModel(Model):
    def __init__(self, engine, controller, alpha, gamma, epsilon, distReward):
        Model.__init__(self, engine, controller, alpha, gamma, epsilon, distReward)

        # Initialize table for all states/actions, initialize to 0
        #self.table = [[[[0 for x in range(self.action_space)] for x in range(self.state_space[2])] for x in range(self.state_space[1])] for x in range(self.state_space[0])]
        self.loadTable("2019-11-21_19-58-35_table_sarsa.pkl")
        print("Table dimensions: ", len(self.table), " x ", len(self.table[0]), " x ", len(self.table[0][0]), " x ", len(self.table[0][0][0]))
        
    def getAction(self, state):
        actions = self.getActionsForState(state)
        return self.getActionEGreedy(actions)
        
    def getActionsForState(self, state):
        return self.table[state[0]][state[1]][state[2]]
        
    def update(self, reward, actionidx, action_taken_val, next_action_val):
        updt = self.alpha * (reward + self.gamma*next_action_val - action_taken_val)
        self.table[self.state[0]][self.state[1]][self.state[2]][actionidx] += updt
        
    def saveTable(self, filename):
        pickle.dump(self.table, open(filename, "wb"))
    
    def loadTable(self, filename):
        self.table = pickle.load(open(filename, "rb"))
        
class QLearn(TabModel):
    def __init__(self, engine, controller, alpha, gamma, epsilon, distReward):
        TabModel.__init__(self, engine, controller, alpha, gamma, epsilon, distReward)
        
    def runEpisode(self):
        self.shots_taken = 0
        self.state = self.getState()
        basket_made = False
        while basket_made is False:
            action_taken_val, actionidx = self.getAction(self.state)
            action = self.engine.mapActionToVals(actionidx)
            if action.action_type == "shoot":
                basket_made, reward = self.doShoot(action)
                self.shots_taken += 1
            else:
                if self.controller:
                    self.controller.move(action.action_type)
                else:
                    self.engine.move(action.action_type)
                # Reward of -0.1 for movement
                reward = -0.1
            
            new_state = self.getState()
            self.update(reward, actionidx, action_taken_val, max(self.getActionsForState(new_state)))
            self.state = new_state
        return self.shots_taken
                    
class Sarsa(TabModel):
    def __init__(self, engine, controller, alpha, gamma, epsilon, distReward):
        TabModel.__init__(self, engine, controller, alpha, gamma, epsilon, distReward)
        
    def runEpisode(self):
        self.shots_taken = 0
        self.state = self.getState()
        action_taken_val, actionidx = self.getAction(self.state)
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
                # Reward of -0.1 for movement
                reward = -0.1
            
            new_state = self.getState()
            new_action_val, new_actionidx = self.getAction(new_state)
            print("ACTION VALUE:", new_action_val)
            self.update(reward, actionidx, action_taken_val, new_action_val)

            self.state = new_state
            actionidx = new_actionidx
            action_taken_val = new_action_val
        return self.shots_taken

# This is identical to Sarsa, but instead of using an epsilon-greedy policy, it uses a Boltzmann distribution
class BoltzSarsa(Sarsa):
    def __init__(self, engine, controller, alpha, gamma, epsilon, tau, distReward):
        Sarsa.__init__(self, engine, controller, alpha, gamma, epsilon, distReward)
        # Temperature value
        self.tau = tau
        
    def getAction(self, state):
        actions = self.getActionsForState(state)
        weightvec = self.genWeightVec(actions)
        idx = np.random.choice(range(len(actions)), p=weightvec)
        return actions[idx], idx
        
    def genWeightVec(self, actions):
        def expWithTemp(val):
            return exp(val/self.tau)
            
        sumexp = sum(map(expWithTemp, actions))
        return list(map(lambda x:expWithTemp(x)/sumexp, actions))
        
class BoltzQLearn(QLearn):
    def __init__(self, engine, controller, alpha, gamma, epsilon, tau, distReward):
        QLearn.__init__(self, engine, controller, alpha, gamma, epsilon, distReward)
        # Temperature value
        self.tau = tau
        
    def getAction(self, state):
        actions = self.getActionsForState(state)
        weightvec = self.genWeightVec(actions)
        idx = np.random.choice(range(len(actions)), p=weightvec)
        return actions[idx], idx
        
    def genWeightVec(self, actions):
        def expWithTemp(val):
            return exp(val/self.tau)
            
        sumexp = sum(map(expWithTemp, actions))
        return list(map(lambda x:expWithTemp(x)/sumexp, actions))

class RankedSarsa(TabModel):
    def __init__(self, engine, controller, alpha, gamma, epsilon, distReward):
        TabModel.__init__(self, engine, controller, alpha, gamma, epsilon, distReward)
        
    def runEpisode(self):
        self.shots_taken = 0
        self.state = self.getState()
        # Initialize a temporary action list for this episode
        actionlist = self.getActionsForState(self.state)
        action_taken_val, actionidx = self.getAction(actionlist)
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
                # Reward of -0.1 for movement
                reward = -0.1
            
            # If the attempted shot failed, give it an infinite negative reward so it will not be tried again for this episode
            actionlist[actionidx] = float("-inf")
            new_action_val, new_actionidx = self.getAction(actionlist)
            self.update(reward, actionidx, action_taken_val, new_action_val)

            actionidx = new_actionidx
            action_taken_val = new_action_val
        return self.shots_taken
        
    def getAction(self, actions):
        return self.getActionEGreedy(actions)
    
class Learner():
    def __init__(self, cfgname, model_type, rendered):
        if not os.path.isfile(cfgname):
            print(cfgname + " does not exist, creating...")
            self.genDefaultCfg()
    
        config = configparser.ConfigParser()
        config.read(cfgname)
        alpha, gamma, epsilon, tau, distReward = self.parseCfg(config['Learner'])
        self.model_type = model_type
        # Whether or not training should be rendered
        self.rendered = rendered
        controller = None
        if self.rendered:
            from physengine import render
            engine = render.PhysWin(config['Game'])
            controller = render.PhysWinController()
        else:
            engine = norender.PymunkSpaceNoRender(config['Game'])
        
        
        if self.model_type == "qlearn":
            self.model = QLearn(engine, controller, alpha, gamma, epsilon, distReward)
        elif self.model_type == "sarsa":
            self.model = Sarsa(engine, controller, alpha, gamma, epsilon, distReward)
        elif self.model_type == "bsarsa":
            self.model = BoltzSarsa(engine, controller, alpha, gamma, epsilon, tau, distReward)
        elif self.model_type == "bqlearn":
            self.model = BoltzQLearn(engine, controller, alpha, gamma, epsilon, tau, distReward)
        elif self.model_type == "rsarsa":
            self.model = RankedSarsa(engine, controller, alpha, gamma, epsilon, distReward)
        elif self.model_type == "sgsarsa":
            self.model = SGSarsa(engine, controller, alpha, gamma, epsilon, distReward)

        
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
        graphfilename = dirname + "/data/" + ts + "_graph_" + self.model_type + ".pkl"
        tablefilename = dirname + "/data/" + ts + "_table_" + self.model_type + ".pkl"
        print("Saving " + graphfilename + "...")
        pickle.dump(shots_taken_graph, open(graphfilename, "wb"))
        tablename = 'null'
        if self.model_type != "sgsarsa":
            print("Saving " + tablefilename + "...")
            self.model.saveTable(tablefilename)
        return graphfilename, tablefilename

        
    def getAction(self):
        return self.model.getAction(bball_x, bball_y, net_x, net_y)
        
    def genDefaultCfg(self):
        config = configparser.ConfigParser()
        config['Learner'] = {'alpha': '0.05',
                                'gamma': '0.9',
                                'epsilon': '0.1',
                                'tau': '0.1',
                                'distReward': 'no'}
                                
        config['Game'] = {'winwidth': '500',
                            'winheight': '500',
                            'CoordStepSize': '50',
                            'MinShotVel': '100',
                            'MaxShotVel': '1000',
                            'VelocityStepSize': '20',
                            'MinShotAngle': '0',
                            'MaxShotAngle': '90',
                            'AngleStepSize': '5',
                            'WindForce': '150',
                            'DefaultGravity': '-900',
                            'AlwaysRandomizeWind': 'no'}
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
            
    def parseCfg(self, cfg):
        alpha = float(cfg['alpha'])
        gamma = float(cfg['gamma'])
        epsilon = float(cfg['epsilon'])
        tau = float(cfg['tau'])
        distReward = cfg.getboolean('distReward')
        return alpha, gamma, epsilon, tau, distReward

    
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
    parser.add_argument("--algo", type=str, default="sarsa", choices=["qlearn", "sarsa", "bsarsa", "bqlearn", "rsarsa", "sgsarsa"], help="Which RL algorithm to use")
    parser.add_argument("--render", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether or not to render the game")
    parser.add_argument("--cfg", type=str, default="config.ini", help="Name of the config file to load")

    args = parser.parse_args()
    graph = args.graph
    eps = args.eps
    algo = args.algo
    rendered = args.render
    cfg = args.cfg

    random.seed(3)
    learner = Learner(cfg, algo, rendered)
    learner.start(eps, graph)
