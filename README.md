# bball_learning
## Description
Reinforcement learning method(s) for a simple 2D basketball shooting game

## Usage
Use `python Learner.py` with any combination of the following flags:
* `--algo [algo_name=qlearn,sarsa]` - which RL algorithm to use (default is sarsa)
* `--eps` - how many episodes to run (default is 1000)
* `--graph` - whether or not to display the graph at the end of running (however a png is always saved with or wihout this flag)
* `--render` - whether or not to render training - enabling this makes training glacially slow
