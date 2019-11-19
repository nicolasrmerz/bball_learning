import os
import matplotlib.pyplot as plt
import pickle
import argparse

dirname = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="Filename")
args = parser.parse_args()
filename = dirname + "/data/" + args.name

shots = pickle.load(open(filename, "rb"))
eps = range(len(shots))

fig, ax = plt.subplots()
ax.plot(eps, shots)

ax.set(xlabel='Episode', ylabel='Shots Taken',
       title='Shots Taken Per Episode')
ax.grid()

fig.savefig(filename + ".png")
plt.show()