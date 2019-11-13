from physengine import render
from physengine import norender

import threading
import pyglet
import time
import random

# TODO
# Currently a stub
class RLModel():
    def __init__(self, model_type):
        # This will be used to select which algorithm to use
        self.model_type = model_type
        
    def getAction(self, bball_x, bball_y, net_x, net_y):
        velocity = random.randrange(200, 1000, 10)
        angle = random.randrange(0, 60, 5)
        # For now, always return random shoot action
        return norender.Action("shoot", velocity, angle)
        
    def update(self, made_basket):
        pass

class Learner():
    def __init__(self, winwidth, winheight, rendered):
        # Whether or not training should be rendered
        self.rendered = rendered
        self.dt = 0.02
        if self.rendered:
            self.engine = render.PhysWin(winwidth, winheight)
        else:
            self.engine = norender.PymunkSpaceNoRender(winwidth, winheight)
        
        # TODO: Replace this statement with an actual model
        self.model = RLModel(0)
        
    def start(self):
        if self.rendered:
            thread = threading.Thread(target = self.control_rendered)
            thread.start()
            pyglet.app.run()

        else:
            while True:
                action = self.getAction()
                self.doActionNoRendered(action)
        
    def control_rendered(self):
        # This is blocking, so the thread will wait until the Physwin is done initializing
        render.done_queue.get()
        while True:
            action = self.getAction()
            render.job_queue.put(action)
            print(render.job_queue.qsize())
            done = render.done_queue.get()
            if action.action_type == "shoot":
                if done:
                    print("Basket made!")
                else:
                    print("Basket missed!")
                self.model.update(done)
            
        
    def doActionNoRendered(self, action):
        if action.action_type == "shoot":
            made_basket = self.engine.shoot(action, self.dt)
            if made_basket:
                print("Basket made!")
            else:
                print("Basket missed!")
            self.model.update(made_basket)
        elif action.action_type == "r":
            print("Moving right")
            self.engine.move("r")
        elif action.action_type == "l":
            print("Moving left")
            self.engine.move("l")        
            
    def getAction(self):
        bball_x, bball_y, net_x, net_y = self.engine.getCoords()
        return self.model.getAction(bball_x, bball_y, net_x, net_y)
        
    
if __name__ == "__main__":
    learner = Learner(500, 600, True)
    learner.start()
