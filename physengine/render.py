import pyglet
from pymunk.pyglet_util import DrawOptions
import time
import random
import threading
import queue
import os
from math import degrees

from . import norender

job_queue = queue.Queue()
done_queue = queue.Queue()
dirname = os.path.dirname(os.path.abspath(__file__))

class PhysWin(pyglet.window.Window):
    def __init__ (self, winwidth, winheight):
        self.winheight = winheight
        self.winwidth = winwidth
        pyglet.window.Window.__init__(self, self.winheight, self.winwidth, fullscreen = False)
        self.options = DrawOptions()
        self.pmSpace = norender.PymunkSpace(self.winwidth, self.winheight)
        background = pyglet.image.SolidColorImagePattern((255,255,255,255)).create_image(self.winheight, self.winwidth)
        self.background_sprite = pyglet.sprite.Sprite(background, x=0, y=0)
        pyglet.clock.schedule_interval(self.update, 1.0/60)
        self.doingAction = False

        hoop_img = pyglet.image.load(dirname + "/../resource/hoop_resized.png")
        hoop_img.anchor_x = hoop_img.width // 2 - 1
        hoop_img.anchor_y = hoop_img.height // 2 - 1
        self.hoop_sprite = pyglet.sprite.Sprite(hoop_img, x=0, y=0)
        
        self.reset_space(True)
        
        basketball_img = pyglet.image.load(dirname + "/../resource/basketball_resized.png")
        basketball_img.anchor_x = basketball_img.width // 2 - 1
        basketball_img.anchor_y = basketball_img.height // 2 - 1
        self.basketball_sprite = pyglet.sprite.Sprite(basketball_img, x=self.pmSpace.basketball_body.position.x, y=self.pmSpace.basketball_body.position.y)
        done_queue.put(True)
        
    def reset_space(self, doReset):
        self.pmSpace.reset_space(doReset)
        self.hoop_sprite.update(self.pmSpace.netx-1, self.pmSpace.nety - 1 - 1.5*self.pmSpace.ball_radius)
        
        
    def check_bounds(self):
        self.pmSpace.check_bounds()

            
    def on_draw(self):
        self.clear()
        self.background_sprite.draw()
        self.pmSpace.space.debug_draw(self.options)
        self.basketball_sprite.draw()
        self.hoop_sprite.draw()
        
    def update(self, dt):
        if not job_queue.empty() and not self.doingAction:
            item = job_queue.get(block=False)
            self.doingAction = True
            if item.action_type == "shoot":
                self.shoot(item)
            elif item.action_type == "r":
                self.move("r")
                done_queue.put(True)
            elif item.action_type == "l":
                self.move("l")
                done_queue.put(True)

        self.pmSpace.space.step(dt)
        self.check_bounds()
        # Have a check to make sure the ball's not stuck
        if self.doingAction:
            self.pmSpace.steps_taken += 1
            if self.pmSpace.steps_taken > self.pmSpace.MAX_STEPS:
                self.reset_space(False)
                self.doingAction = False
                done_queue.put(False)
        self.basketball_sprite.update(self.pmSpace.basketball_body.position.x, self.pmSpace.basketball_body.position.y, degrees(-self.pmSpace.basketball_body.angle))
        if self.pmSpace.ball_hit:
            self.reset_space(True)
            self.doingAction = False
            done_queue.put(True)
        if self.pmSpace.ball_missed:
            # Reset the space but don't move the position of the ball or the hoop
            self.reset_space(False)
            self.doingAction = False
            done_queue.put(False)
            
    def shoot(self, action):
        self.pmSpace.shoot(action)
            
    def move(self, dir):
        self.pmSpace.move(dir)
        self.doingAction = False
        
    def getCoords(self):
        return self.pmSpace.getCoords()
        
    def getStateSpace(self):
        return self.pmSpace.getStateSpace()
        
    def getActionSpace(self):
        return self.pmSpace.getActionSpace()
            
    def mapValsToState(self, net_x, net_y, ball_x):
        return self.pmSpace.mapValsToState(net_x, net_y, ball_x)
        
    def mapValsToAction(self, action):
        return self.pmSpace.mapValsToAction(action)
    
    def mapActionToVals(self, actionidx):
        return self.pmSpace.mapActionToVals(actionidx)

def test():
    time.sleep(2)
    job_queue.put(Action("r"))
    job_queue.put(Action("r"))
    job_queue.put(Action("r"))
    job_queue.put(Action("r"))
    
    job_queue.put(Action("shoot", 600, 45))

        

if __name__ == "__main__":
    thread = threading.Thread(target = test)
    thread.start()
    window = PhysWin(500, 600)
    pyglet.app.run()
