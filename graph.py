import os
import matplotlib.pyplot as plt
import pickle
import argparse

dirname = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="Filename")
parser.add_argument("--range", type=int, nargs=2, help="Range of the table to plot")
args = parser.parse_args()
filename = dirname + "/data/" + args.name

shots = pickle.load(open(filename, "rb"))

fig, ax = plt.subplots()
if args.range:
    rangemin = args.range[0]
    rangemax = args.range[1]
    ax.set_xlim([rangemin, rangemax])
else:
    rangemin = 0
    rangemax = len(shots)

ax.set_ylim([0, max(shots[rangemin:rangemax])])
#eps = range(rangemin, rangemax)
eps = range(len(shots))

ax.plot(eps, shots)

#ax.set_ylim([0, shotsmax])

ax.set(xlabel='Episode', ylabel='Shots Taken',
       title='Shots Taken Per Episode')
ax.grid()

fig.savefig(filename + ".png")
plt.show()
