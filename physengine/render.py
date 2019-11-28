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

# The intention of this controller is for it to be run in a thread, and send/receive instructions to/from the job/done queue, respectively
class PhysWinController():
    def __init__ (self):
        # This is a blocked get from the queue; execution will not progress until something is in the done queue.
        # Since the game engine (when rendered) places something in the done_queue when it is done intializing, the effect is that the controller will wait until 
        # the environment is done being initialized
        done_queue.get()
        
    # Put a shoot action into the job_queue, and receive the missed_by value
    def shoot(self, action):
        job_queue.put(action)
        missed_by = done_queue.get()
        return missed_by
    
    # Tell the agent to move, and wait till it is done
    def move(self, dir):
        job_queue.put(norender.Action(dir, 0, 0))
        done = done_queue.get()

class PhysWin(pyglet.window.Window):
    def __init__ (self, cfg):
        # Set the window width and height from the config file
        self.winwidth = int(cfg['winwidth'])
        self.winheight = int(cfg['winheight'])
        # Boilerplate pyglet stuff, intialize the environment
        pyglet.window.Window.__init__(self, self.winheight, self.winwidth, fullscreen = False)
        self.options = DrawOptions()
        # Create the underlying pymunk engine
        self.pmSpace = norender.PymunkSpace(cfg)
        # Solid white background
        background = pyglet.image.SolidColorImagePattern((255,255,255,255)).create_image(self.winheight, self.winwidth)
        self.background_sprite = pyglet.sprite.Sprite(background, x=0, y=0)
        # At every 1.0/60 interval, call the update function (essentially render at 60 FPS)
        pyglet.clock.schedule_interval(self.update, 1.0/60)
        # This keeps track of if the agent is in the middle of a shot; the engine will not take another job off the job_queue until it finishes whatever it is doing
        self.doingAction = False
        
        # Load the net and ball images
        hoop_img = pyglet.image.load(dirname + "/../resource/hoop_resized.png")
        hoop_img.anchor_x = hoop_img.width // 2 - 1
        hoop_img.anchor_y = hoop_img.height // 2 - 1
        self.hoop_sprite = pyglet.sprite.Sprite(hoop_img, x=0, y=0)
        
        # Reset the pymunk space
        self.reset_space(True)
        
        basketball_img = pyglet.image.load(dirname + "/../resource/basketball_resized.png")
        basketball_img.anchor_x = basketball_img.width // 2 - 1
        basketball_img.anchor_y = basketball_img.height // 2 - 1
        self.basketball_sprite = pyglet.sprite.Sprite(basketball_img, x=self.pmSpace.basketball_body.position.x, y=self.pmSpace.basketball_body.position.y)
        done_queue.put(True)
        
    # Reset the pymunk environment and reset the position of the hoop sprite
    def reset_space(self, doReset):
        self.pmSpace.reset_space(doReset)
        self.hoop_sprite.update(self.pmSpace.netx-1, self.pmSpace.nety - 1 - 1.5*self.pmSpace.ball_radius)
        
    # Use pymunk to check if the ball has made it into the net - see norender's check_bounds for more details
    def check_bounds(self):
        self.pmSpace.check_bounds()

    # What to render every frame
    def on_draw(self):
        self.clear()
        self.background_sprite.draw()
        self.pmSpace.space.debug_draw(self.options)
        self.basketball_sprite.draw()
        self.hoop_sprite.draw()
        
    # This is called 60 times per second (if the computer is not overloaded and is running at full speed)
    def update(self, dt):
        # If there is something in the job_queue and the environment is not currently in the middle of an action
        if not job_queue.empty() and not self.doingAction:
            # Get a task from the job_queue
            item = job_queue.get(block=False)
            # Make sure to set the flag that the agent is doing something
            self.doingAction = True
            if item.action_type == "shoot":
                self.shoot(item)
            elif item.action_type == "r":
                self.move("r")
                done_queue.put(True)
            elif item.action_type == "l":
                self.move("l")
                done_queue.put(True)
        
        # Step pymunk one unit forward in time
        self.pmSpace.space.step(dt)
        # Every update, check if the ball is within the bounds of the net
        self.check_bounds()
        # Have a check to make sure the ball's not stuck
        if self.doingAction:
            self.pmSpace.steps_taken += 1
            if self.pmSpace.steps_taken > self.pmSpace.MAX_STEPS:
                self.reset_space(False)
                self.doingAction = False
                done_queue.put(1)
        # Update the position of the basketball sprite to match the pymunk basketball
        self.basketball_sprite.update(self.pmSpace.basketball_body.position.x, self.pmSpace.basketball_body.position.y, degrees(-self.pmSpace.basketball_body.angle))
        # If the ball is made in
        if self.pmSpace.ball_hit:
            self.reset_space(True)
            self.doingAction = False
            # Put a 0 to mean that the basket was made
            done_queue.put(0)
        #
        if self.pmSpace.ball_missed:
            # Reset the space but don't move the position of the ball or the hoop
            missed_by = self.pmSpace.ball_missed_by
            self.reset_space(False)
            self.doingAction = False
            # Put how much the agent missed the shot by
            done_queue.put(missed_by)
            
    def shoot(self, action):
        self.pmSpace.shoot(action)
        
    # There is no waiting needed for a move, just move the balls position and set doingAction to done
    def move(self, dir):
        self.pmSpace.move(dir)
        self.doingAction = False
        
    # These next functions just call the equivalent function from its pymunk engine - this is so Learner.py does not have to know what engine it has and can just call these functions
    def getMissedBy(self):
        return self.pmSpace.getMissedBy()
            
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
