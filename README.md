# bball_learning
## Description
Reinforcement learning method(s) for a simple 2D basketball shooting game

## Usage - Learner.py
Use `python Learner.py` with any combination of the following flags:
* `--algo [algo_name=qlearn,sarsa]` - which RL algorithm to use (default is sarsa)
* `--eps` - how many episodes to run (default is 1000)
* `--graph` - whether or not to display the graph at the end of running (however a png is always saved with or wihout this flag)
* `--render` - whether or not to render training - enabling this makes training glacially slow
* `--cfg` - the name of the config file for the script to use (default is config.ini)
* `--table` - the name of the pickle file which contains a pre-trained table for the script to use (default creates a new table)

## Usage - graph.py
graph.py is a utility to generate a matplotlib graph using a graph pickle file
Use `python graph.py` with the following flags:
* The filename of the pickle file MUST immediately be supplied as the first argument, and should appear in the root directory
* `--range` - the range of episodes to graph in matplotlib (default plots every episode)
