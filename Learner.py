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

# This is the class for the base model object that will be used by Learner. It will hold the value of alpha, gamma, epsilon, and the boolean of whether or not to use the flat or proportional rewards.
# It also contains the engine that it will be applying actions to, and the controller for the rendered window, if the script was passed the --render flag
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
        
    # Get the current coordinates of the net and ball from the engine
    def getState(self):
        net_x, net_y, ball_x = self.engine.getCoords()
        return self.engine.mapValsToState(net_x, net_y, ball_x)
        
    # Action selection for epsilon greedy
    def getActionEGreedy(self, actions):
        # Epsilon greedy; if generated value is less than epsilon, randomly pick an action, otherwise take the max action
        if random.uniform(0, 1) <= self.epsilon:
            idx = random.randrange(0, len(actions))
            return actions[idx], idx
        else:
            return self.getMaxAction(actions)
    
    # Returns the max value and its index from the supplied action list
    def getMaxAction(self, actions):
        max_value = max(actions)
        return max_value, actions.index(max_value)
        
    # Tells the engine to take a shot (using the action supplied to this method),
    # and returns whether the shot was made and the reward
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
        
# This is the class for a generic function approximation method that uses a neural network. On top of the attributes of model, it contains the neural network itself
# The neural network has 2 hidden layers, with 200 and 100 nodes respectively
class ApproxModel(Model):
    def __init__(self, engine, controller, alpha, gamma, epsilon, distReward):
        Model.__init__(self, engine, controller, alpha, gamma, epsilon, distReward)

        self.nn = Sequential()
        self.nn.add(Dense(200, batch_input_shape=(1,len(self.state_space)), activation='sigmoid'))
        self.nn.add(Dense(100, activation='sigmoid'))
        self.nn.add(Dense(self.action_space, activation='linear'))
        opt = Adam(learning_rate=self.alpha)
        self.nn.compile(loss='mse', optimizer=opt, metrics=['mae'])

# This actually implements the semi-gradient SARSA algorithm
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
        

# This class implements the generic tabular model, the functionality of which is extended by the various SARSA and Q-Learning variations as defined in subsequent classes
class TabModel(Model):
    def __init__(self, engine, controller, alpha, gamma, epsilon, distReward, table):
        Model.__init__(self, engine, controller, alpha, gamma, epsilon, distReward)

        # Initialize table for all states/actions, initialize to 0, or load a table
        if table:
            self.loadTable(table)
        else:
            self.table = [[[[0 for x in range(self.action_space)] for x in range(self.state_space[2])] for x in range(self.state_space[1])] for x in range(self.state_space[0])]
        
        print("Table dimensions: ", len(self.table), " x ", len(self.table[0]), " x ", len(self.table[0][0]), " x ", len(self.table[0][0][0]))
        
    # Uses epsilon-greedy to select an action given a state
    def getAction(self, state):
        actions = self.getActionsForState(state)
        return self.getActionEGreedy(actions)
        
    # Get the set of actions available at the given state
    def getActionsForState(self, state):
        return self.table[state[0]][state[1]][state[2]]
        
    # This is a generic update rule (SARSA will use the actual next action as the action_taken_val, whereas Q-Learning will supply the max of the next set of actions)
    def update(self, reward, actionidx, action_taken_val, next_action_val):
        updt = self.alpha * (reward + self.gamma*next_action_val - action_taken_val)
        self.table[self.state[0]][self.state[1]][self.state[2]][actionidx] += updt
        
    # Save the table with the given filename
    def saveTable(self, filename):
        pickle.dump(self.table, open(filename, "wb"))
    
    # Load a table with the given filename
    def loadTable(self, filename):
        self.table = pickle.load(open(filename, "rb"))
        
# The purpose of this extended class is to implement the Q-Learning algorithm, exactly as found in the "Reinforcement Learning" textbook
class QLearn(TabModel):
    def __init__(self, engine, controller, alpha, gamma, epsilon, distReward, table):
        TabModel.__init__(self, engine, controller, alpha, gamma, epsilon, distReward, table)
        
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
             
# The purpose of this extended class is to implement the SARSA algorithm, exactly as found in the "Reinforcement Learning" textbook             
class Sarsa(TabModel):
    def __init__(self, engine, controller, alpha, gamma, epsilon, distReward, table):
        TabModel.__init__(self, engine, controller, alpha, gamma, epsilon, distReward, table)
        
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
            self.update(reward, actionidx, action_taken_val, new_action_val)

            self.state = new_state
            actionidx = new_actionidx
            action_taken_val = new_action_val
        return self.shots_taken

# This is identical to SARSA, but instead of using an epsilon-greedy policy, it uses a Boltzmann distribution
class BoltzSarsa(Sarsa):
    def __init__(self, engine, controller, alpha, gamma, epsilon, tau, distReward, table):
        Sarsa.__init__(self, engine, controller, alpha, gamma, epsilon, distReward, table)
        # Temperature value
        self.tau = tau
        
    # Gets the set of actions, then weights their choice based on the Boltzmann Distribution    
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
        
# This class is identical to BoltzSarsa, but instead extends QLearn
class BoltzQLearn(QLearn):
    def __init__(self, engine, controller, alpha, gamma, epsilon, tau, distReward, table):
        QLearn.__init__(self, engine, controller, alpha, gamma, epsilon, distReward, table)
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
    
# Main learner class that starts the learning process.
# Takes in the name of the config file, which algorithm to use, whether or not to render the game, and a table filename (if one is supplied)
class Learner():
    def __init__(self, cfgname, model_type, rendered, table):
        # If the supplied config name is not found, it is created with default values
        if not os.path.isfile(cfgname):
            print(cfgname + " does not exist, creating...")
            self.genDefaultCfg()
    
        config = configparser.ConfigParser()
        config.read(cfgname)
        # Take these parameters from the "Learning" section of the config file
        alpha, gamma, epsilon, tau, distReward = self.parseCfg(config['Learner'])
        self.model_type = model_type
        # Whether or not training should be rendered
        self.rendered = rendered
        controller = None
        # If it is rendered, a controller that runs in a seperate thread is used, which is initialized here
        if self.rendered:
            # Do the import here, as pyglet will complain if it is imported at the top no matter what over command line (say you want to ssh into a server and run it un-rendered, it will not work)
            from physengine import render
            engine = render.PhysWin(config['Game'])
            controller = render.PhysWinController()
        # Otherwise, use the non-rendered pymunk engine
        else:
            engine = norender.PymunkSpaceNoRender(config['Game'])
        
        if self.model_type == "qlearn":
            self.model = QLearn(engine, controller, alpha, gamma, epsilon, distReward, table)
        elif self.model_type == "sarsa":
            self.model = Sarsa(engine, controller, alpha, gamma, epsilon, distReward, table)
        elif self.model_type == "bsarsa":
            self.model = BoltzSarsa(engine, controller, alpha, gamma, epsilon, tau, distReward, table)
        elif self.model_type == "bqlearn":
            self.model = BoltzQLearn(engine, controller, alpha, gamma, epsilon, tau, distReward, table)
        elif self.model_type == "sgsarsa":
            self.model = SGSarsa(engine, controller, alpha, gamma, epsilon, distReward)

    # Start the control method, which runs the episodes
    def start(self, episodes, graph):
        self.graph = graph
        if self.rendered:
            thread = threading.Thread(target = self.control, args=([episodes]))
            thread.start()
            pyglet.app.run()

        else:
            self.control(episodes)
        
    # Run whichever algorithm has been set to self.model for the supplied number of episodes
    def control(self, episodes):
        shots_taken_graph = []
        pbar = tqdm(total=episodes)
        for i in range(episodes):
            # Each 100 episodes, update the progress bar, as doing it more frequently actually negatively effects performance
            if i % 100 is 0:
                pbar.update(100)
            shots_taken = self.model.runEpisode()
            shots_taken_graph.append(shots_taken)
        pbar.close()
        # Convert the graph list and the table to binaries and save them
        graphfilename, tablefilename = self.saveListAndTable(shots_taken_graph)
        self.plotGraph(shots_taken_graph, graphfilename)
        
    # Save a plot of the episodes. If the --graph flag was passed, matplotlib will actually open
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
    
    #  Save the graph and table
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
        
    # This generates a default config file. These values were used for all the tests performed.
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
            
    # This is intended to parse the "Learner" section of the config file
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
            
    # Parse any/all of the supplied flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether or not to display the graph of the results")
    parser.add_argument("--eps", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--algo", type=str, default="sarsa", choices=["qlearn", "sarsa", "bsarsa", "bqlearn", "sgsarsa"], help="Which RL algorithm to use")
    parser.add_argument("--render", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether or not to render the game")
    parser.add_argument("--cfg", type=str, default="config.ini", help="Name of the config file to load")
    parser.add_argument("--table", type=str, default=None, help="Name of the table file to load")


    args = parser.parse_args()
    graph = args.graph
    eps = args.eps
    algo = args.algo
    rendered = args.render
    cfg = args.cfg
    table = args.table

    # Tests were done using a set seed
    random.seed(3)
    learner = Learner(cfg, algo, rendered, table)
    learner.start(eps, graph)
